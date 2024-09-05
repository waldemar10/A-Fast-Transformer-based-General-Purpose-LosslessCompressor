import collections
import gzip
import os
import time
import utils
import struct
import psutil
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import shutil
import GPUtil
import threading

import compress_model
import arithmeticcoding_fast

from absl import app
from absl import flags
from absl import logging

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.set_printoptions(profile="full") 
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('batch_size', 512, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Adam Optimizer learning rate.')
flags.DEFINE_integer('hidden_dim', 256, 'Feature dimension.')
flags.DEFINE_integer('vocab_dim', 64, 'Feature dimension.')
flags.DEFINE_integer('n_layers', 1, 'Number of Attention layers.')
flags.DEFINE_integer('ffn_dim', 4096, 'MLP dimension in model.')
flags.DEFINE_integer('n_heads', 8, 'Number of heads for attention.')
flags.DEFINE_string(
    'feature_type', 'sqr',
    'Nonlinearity function for feature. Can be relu, elu+1, sqr, favor+, or favor+{int}.'
)
flags.DEFINE_enum(
    'compute_type', 'iter', ['iter', 'ps', 'parallel_ps'],
    'Which type of method to compute: iter = iterative algorithm, ps = implementation using torch.cumsum, parallel_ps = implementation using custom log prefix sum implementation.'
)
flags.DEFINE_float('weight_decay', 0.0, 'Weight decay for regularization.')

# Training parameters
flags.DEFINE_string('gpu_id', '0', 'ID of GPU.')
flags.DEFINE_integer('random_seed', 0, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('print_step', 1000, 'Interval to print metrics.')
# Dataset parameters
flags.DEFINE_integer('seq_len', 8, 'Maximum sequence length (L).')
flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size of data.')
flags.DEFINE_string('input_dir', 'aaa', 'input data dir')
flags.DEFINE_string('prefix', 'text8', 'output dir')

def setup(rank, world_size):
    # Initialisiere den Prozess
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def get_gpu_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2, torch.cuda.memory_reserved() / 1024**2
    return 0, 0

def monitor_resources(cpu_usages, memory_usages, gpu_usages, stop_event):
    while not stop_event.is_set():
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_usage_mb = memory_info.used / (1024 * 1024)

        gpus = GPUtil.getGPUs()
        gpu_percent = max([gpu.load * 100 for gpu in gpus]) if gpus else 0.0

        cpu_usages.append(cpu_percent)
        memory_usages.append(memory_usage_mb)
        gpu_usages.append(gpu_percent)

        time.sleep(0.5)

def log_resource_usage(start_time, phase, file_path, original_size=None, compressed_size=None, 
                       cpu_usage=None, memory_usage=None, gpu_usage=None):
    end_time = time.time()
    elapsed_time = end_time - start_time

    cpu_usage = cpu_usage if cpu_usage is not None else psutil.cpu_percent()
    mem_usage = memory_usage if memory_usage is not None else psutil.virtual_memory().percent

    with open(file_path, 'a', encoding='utf-8') as f:
        if original_size is not None and compressed_size is not None:
            compression_ratio = original_size / compressed_size
            f.write(f"Original File Size: {original_size / (1024 * 1024):.2f} MB\n")
            f.write(f"Compressed File Size: {compressed_size / (1024 * 1024):.2f} MB\n")
            f.write(f"Compression Ratio: {compression_ratio:.2f}\n")
        f.write(f"{phase} Time: {elapsed_time:.2f} seconds\n")
        f.write(f"{phase} CPU Usage: {cpu_usage:.2f} %\n")
        f.write(f"{phase} Memory Usage: {mem_usage:.2f} MB\n")
        if gpu_usage:
            f.write(f"{phase} GPU Usage: {gpu_usage:.2f} %\n")
        f.write("\n")

def decode(temp_dir, compressed_file, FLAGS, len_series, last):
  cpu_usages, memory_usages, gpu_usages = [], [], []
  stop_event = threading.Event()
  monitor_thread = threading.Thread(target=monitor_resources, args=(cpu_usages, memory_usages, gpu_usages, stop_event))
  monitor_thread.start()
  start_time = time.time()
  bs = FLAGS.batch_size

  iter_num = (len_series - FLAGS.seq_len) // FLAGS.batch_size
  
  ind = np.array(range(bs))*iter_num
  print(iter_num - FLAGS.seq_len)
  series_2d = np.zeros((bs,iter_num), dtype = np.uint8).astype('int')

  f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'rb') for i in range(bs)]
  bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
  dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

  prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
  cumul = np.zeros(FLAGS.vocab_size+1, dtype = np.uint64)
  cumul[1:] = np.cumsum(prob*10000000 + 1)

  # Decode first K symbols in each stream with uniform probabilities
  for i in range(bs):
    for j in range(min(FLAGS.seq_len, iter_num)):
      series_2d[i,j] = dec[i].read(cumul, FLAGS.vocab_size)
  
  cumul_batch = np.zeros((bs, FLAGS.vocab_size+1), dtype = np.uint64)

  """ os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id """
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  model = compress_model.SLiMPerformer(FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,FLAGS.n_layers, FLAGS.ffn_dim,FLAGS.n_heads, FLAGS.feature_type, FLAGS.compute_type).cuda()
  print(model)
  original_model = model
  model = DDP(model, device_ids=[0])
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))

  training_start = time.time()
  for train_index in range(iter_num-FLAGS.seq_len):
    model.train()
    train_batch = torch.LongTensor(series_2d[:, train_index:train_index + FLAGS.seq_len]).cuda()
    logits = model.forward(train_batch)
    prob = logits[:, -1, :]
    prob = F.softmax(prob, dim=1).detach().cpu().numpy()
    
    cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)

    # Decode with Arithmetic Encoder
    for i in range(bs):
      series_2d[i,train_index+FLAGS.seq_len] = dec[i].read(cumul_batch[i,:], FLAGS.vocab_size)
    
    logits = logits.transpose(1, 2)
    label = torch.from_numpy(series_2d[:, train_index+1:train_index+FLAGS.seq_len+1]).cuda()
    label = label.type(torch.LongTensor).cuda()
    train_loss = torch.nn.functional.cross_entropy(logits[:, :, -1], label[:, -1], reduction='mean')
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    if train_index % FLAGS.print_step == 0:
      print(train_index, ":", train_loss.item()/np.log(2))
  
    
  out = open('tttdecompressed_out', 'w', encoding='utf-8')
  for i in range(len(series_2d)):
    out.write(utils.decode_tokens(series_2d[i]).encode('utf-8', errors='ignore').decode('utf-8'))
  
  
  for i in range(bs):
    bitin[i].close()
    f[i].close()

  if last:
    series = np.zeros(last, dtype = np.uint8).astype('int')
    f = open(temp_dir+"/"+compressed_file+'.last','rb')
    bitin = arithmeticcoding_fast.BitInputStream(f)
    dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
    prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)

    for j in range(last):
      series[j] = dec.read(cumul, FLAGS.vocab_size)
  
    print("Last decode part don't need inference.")
    out.write(utils.decode_tokens(series))
    print(utils.decode_tokens(series))
    bitin.close()
    f.close()

    stop_event.set()
    monitor_thread.join()

    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
    avg_gpu_usage = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0

    log_resource_usage(start_time, "Decode", "analysis.txt",original_size=None, compressed_size=None, cpu_usage=avg_cpu_usage,
                        memory_usage=avg_memory_usage, gpu_usage=avg_gpu_usage)
    return
 

def encode(temp_dir, compressed_file, FLAGS, series, train_data, last_train_data):
    cpu_usages, memory_usages, gpu_usages = [], [], []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(cpu_usages, memory_usages, gpu_usages, stop_event))
    monitor_thread.start()
    start_time = time.time()
    bs = FLAGS.batch_size

    f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'wb') for i in range(bs)]
    bitout = [arithmeticcoding_fast.BitOutputStream(f[i]) for i in range(bs)]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]
    
    prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)
    
    iter_num = len(train_data) // FLAGS.batch_size
    ind = np.array(range(bs))*iter_num
    iter_num -= FLAGS.seq_len

    for i in range(bs):
        for j in range(FLAGS.seq_len):
            enc[i].write(cumul, series[ind[i]+j])
    
    cumul_batch = np.zeros((bs, FLAGS.vocab_size+1), dtype = np.uint64)

    model = compress_model.SLiMPerformer(FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,
                                             FLAGS.n_layers, FLAGS.ffn_dim,
                                             FLAGS.n_heads, FLAGS.feature_type, FLAGS.compute_type).cuda()
    print(model)
    original_model = model
    model = DDP(model, device_ids=[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
    print(iter_num)
    for train_index in range(iter_num):
        model.train()
        train_batch = train_data[ind, :]
        y = train_batch[:, -1]
        train_batch = torch.from_numpy(train_batch).cuda().long()
        
        train_loss, logits = original_model.full_loss(train_batch, with_grad=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        logits = logits.transpose(1, 2)
        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()
        cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)
        
        for i in range(bs):
            enc[i].write(cumul_batch[i,:], y[i])
        
        ind += 1
        if train_index % FLAGS.print_step == 0:
            size = 0
            for cf in os.listdir(temp_dir):
                size += os.path.getsize(temp_dir+"/"+cf)
            print(train_index, ":", train_loss.item()/np.log(2), "size:", size/(1024*1024))
    
    for i in range(bs):
        enc[i].finish()
        bitout[i].close()
        f[i].close()

    if last_train_data is not None:
        print("last series")
        f = open(temp_dir+"/"+compressed_file+'.last','wb')
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
        cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)
    
        for j in range(len(last_train_data)):
            enc.write(cumul, last_train_data[j])
        print("Last encode part don't need inference.")
    
        enc.finish()
        bitout.close()
        f.close()
    
    # Compute the size of the compressed file
    compressed_size = sum(os.path.getsize(temp_dir + "/" + compressed_file + '.' + str(i)) for i in range(bs))
    if last_train_data is not None:
        compressed_size += os.path.getsize(temp_dir + "/" + compressed_file + '.last')
    stop_event.set()
    monitor_thread.join()

    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
    avg_gpu_usage = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0

    input_file_path = FLAGS.input_dir
    original_size = os.path.getsize(input_file_path)
    # Log resource usage
    log_resource_usage(start_time, "Encode", "analysis.txt", original_size=original_size, compressed_size=compressed_size,cpu_usage=avg_cpu_usage,
                        memory_usage=avg_memory_usage, gpu_usage=avg_gpu_usage)
    cleanup()
    return
    
def var_int_encode(byte_str_len, f):
  while True:
    this_byte = byte_str_len&127
    byte_str_len >>= 7
    if byte_str_len == 0:
      f.write(struct.pack('B',this_byte))
      break
    f.write(struct.pack('B',this_byte|128))
    byte_str_len -= 1

def var_int_decode(f):
    byte_str_len = 0
    shift = 1
    while True:
        this_byte = struct.unpack('B', f.read(1))[0]
        byte_str_len += (this_byte & 127) * shift
        if this_byte & 128 == 0:
                break
        shift <<= 7
        byte_str_len += shift
    return byte_str_len

def main(_):
  rank = 0  
  world_size = torch.cuda.device_count() 
  setup(rank, world_size)

  with open("analysis.txt", 'w', encoding='utf-8') as f:
        f.write("Analysis of Compression and Decompression\n")

  """ os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id """
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  temp_dir = "{}_{}_{}_{}_bs{}_{}_seq{}_temp".format(FLAGS.prefix, FLAGS.vocab_dim, FLAGS.hidden_dim, FLAGS.ffn_dim, FLAGS.batch_size, FLAGS.n_layers, FLAGS.seq_len)
  compressed_file = temp_dir.replace("_temp", ".compressed")
  os.mkdir(temp_dir)
  print(compressed_file)
  
  def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))
  
  old_seq_len = FLAGS.seq_len
  FLAGS.seq_len = FLAGS.seq_len*(FLAGS.hidden_dim // FLAGS.vocab_dim)
  print("FLAGS.seq_len change from {} to {} due to FLAGS.vocab_dim = {} and FLAGS.hidden_dim = {}.".format(old_seq_len, FLAGS.seq_len, FLAGS.vocab_dim, FLAGS.hidden_dim))
  
  with open(FLAGS.input_dir, 'rb') as fp:#, encoding='latin-1') as fp:
    series = np.fromstring(fp.read(), dtype=np.uint8)
  train_data = strided_app(series, FLAGS.seq_len+1, 1)
  
  total_length = len(train_data)
  if total_length % FLAGS.batch_size == 0:
    encode(temp_dir, compressed_file, FLAGS, series, train_data, None)
  else:
    l = total_length // FLAGS.batch_size * FLAGS.batch_size
    encode(temp_dir, compressed_file, FLAGS, series[:l+FLAGS.seq_len], train_data[:l], series[l:])

  #Combined compressed results
  f = open(compressed_file+'.combined','wb')
  for i in range(FLAGS.batch_size):
    f_in = open(temp_dir+'/'+compressed_file+'.'+str(i),'rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  
  if total_length % FLAGS.batch_size != 0:
    f_in = open(temp_dir+'/'+compressed_file+'.last','rb')
    byte_str = f_in.read()
    byte_str_len = len(byte_str)
    var_int_encode(byte_str_len, f)
    f.write(byte_str)
    f_in.close()
  f.close()
  
  total = 0
  for ff in os.listdir(temp_dir):
    total += os.path.getsize(temp_dir+'/'+ff)
  
  print(total/(1024*1024))
  
  #Remove temp file
  shutil.rmtree(temp_dir)
  
  #Decode
  os.mkdir(temp_dir)
  
  #Split compressed file
  
  f = open(compressed_file+'.combined','rb')
  len_series = len(series) 
  for i in range(FLAGS.batch_size):
    f_out = open(temp_dir+'/'+compressed_file+'.'+str(i),'wb')
    byte_str_len = var_int_decode(f)
    byte_str = f.read(byte_str_len)
    f_out.write(byte_str)
    f_out.close()
  
  f_out = open(temp_dir+'/'+compressed_file+'.last','wb')
  byte_str_len = var_int_decode(f)
  byte_str = f.read(byte_str_len)
  f_out.write(byte_str)
  f_out.close()
  f.close()
  
  len_series = len(series)
  if (len_series-FLAGS.seq_len) % FLAGS.batch_size == 0:
    decode(temp_dir, compressed_file, FLAGS, len_series, 0)
  else:
    last_length = (len_series - FLAGS.seq_len) % FLAGS.batch_size + FLAGS.seq_len
    decode(temp_dir, compressed_file, FLAGS, len_series, last_length)
  

if __name__ == '__main__':
  app.run(main)
