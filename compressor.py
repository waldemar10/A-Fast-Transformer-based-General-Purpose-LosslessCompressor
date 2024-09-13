import collections
import gzip
import os
import time
import utils
import struct
import psutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import shutil
import GPUtil
import threading
import sys

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

""" os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355' """
os.environ["USE_LIBUV"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def init_distributed_mode(rank,world_size):
    """ print(torch.distributed.is_nccl_available()) """
    """ dist.TCPStore('localhost', '12355', 1, True, use_libuv=False) """
    """ print(f"Number of GPUs available: {torch.cuda.device_count()}") """
    print(f"Current GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    """ print(torch.cuda.device_count()) 
    print(torch.cuda.is_available()) """

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend='nccl',rank=rank, world_size=world_size)
    print(f"Rank: {rank}")
    torch.cuda.set_device(rank)
    print(f"Current GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("Init Process Group")

    """ torch.cuda.set_device(int(FLAGS.gpu_id.split(',')[0])) """

    print("Set Device")

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

def decode(rank,temp_dir, compressed_file, FLAGS, len_series, last):

  start_index = rank * (FLAGS.batch_size // torch.distributed.get_world_size())
  end_index = ((rank + 1) * (FLAGS.batch_size // torch.distributed.get_world_size()))

  cpu_usages, memory_usages, gpu_usages = [], [], []
  stop_event = threading.Event()
  monitor_thread = threading.Thread(target=monitor_resources, args=(cpu_usages, memory_usages, gpu_usages, stop_event))
  monitor_thread.start()
  start_time = time.time()
  bs = FLAGS.batch_size // torch.distributed.get_world_size()

  iter_num = (len_series - FLAGS.seq_len) // FLAGS.batch_size
  """ iter_num = iter_num // world_size """
  print(f"Iterationszahl pro Batch: {iter_num - FLAGS.seq_len}")

  """ ind = np.array(range(bs))*iter_num """
  ind = np.array(range(start_index, end_index)) * iter_num

  series_2d = np.zeros((bs,iter_num), dtype = np.uint8).astype('int')
  print(series_2d)
  """ temp_dir_rank = temp_dir + f"/rank_{rank}_temp" """
  f = [open(temp_dir + "/" + compressed_file + '.' + str(i), 'rb') for i in range(start_index,end_index)]
  bitin = [arithmeticcoding_fast.BitInputStream(f[i-start_index]) for i in range(start_index,end_index)]
  dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i-start_index]) for i in range(start_index,end_index)]

  """ f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'rb') for i in range(bs)]
  bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
  dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)] """

  prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
  cumul = np.zeros(FLAGS.vocab_size+1, dtype = np.uint64)
  cumul[1:] = np.cumsum(prob*10000000 + 1)

  # Decode first K symbols in each stream with uniform probabilities
  for i in range(start_index, end_index):
    for j in range(min(FLAGS.seq_len, iter_num)):
      series_2d[i - start_index, j] = dec[i - start_index].read(cumul, FLAGS.vocab_size)
  
  cumul_batch = np.zeros((end_index - start_index, FLAGS.vocab_size+1), dtype = np.uint64)

  
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  model = compress_model.SLiMPerformer(FLAGS.vocab_size, FLAGS.vocab_dim,
                                        FLAGS.hidden_dim,FLAGS.n_layers, 
                                        FLAGS.ffn_dim,FLAGS.n_heads, FLAGS.feature_type, 
                                        FLAGS.compute_type).cuda()
  print(model)

  try:
      model = DDP(model, device_ids=[rank])
  except Exception as e:
      print(f"DDP Initialization Error on rank {rank}: {e}")
  
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
  
  training_start = time.time()
  print("Decode")
  print(iter_num)
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
  
    
  """ out = open('tttdecompressed_out', 'w', encoding='utf-8')
  for i in range(len(series_2d)):
    out.write(utils.decode_tokens(series_2d[i]).encode('utf-8', errors='ignore').decode('utf-8')) """
  
  output_file = os.path.join(temp_dir, f'decompressed_{rank}.out')
  with open(output_file, 'w', encoding='utf-8') as out:
        for i in range(bs):
            out.write(utils.decode_tokens(series_2d[i]).encode('utf-8', errors='ignore').decode('utf-8'))
  
  
  """ for i in range(bs):
    bitin[i].close()
    f[i].close() """
 
  """ if last:
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
    f.close() """

  if last:
        series = np.zeros(last, dtype=np.uint8).astype('int')
        with open(os.path.join(temp_dir, compressed_file + '.last'), 'rb') as f:
            bitin = arithmeticcoding_fast.BitInputStream(f)
            dec = arithmeticcoding_fast.ArithmeticDecoder(32, bitin)
            prob = np.ones(FLAGS.vocab_size) / FLAGS.vocab_size
            cumul = np.zeros(FLAGS.vocab_size + 1, dtype=np.uint64)
            cumul[1:] = np.cumsum(prob * 10000000 + 1)

            for j in range(last):
                series[j] = dec.read(cumul, FLAGS.vocab_size)

        with open(output_file, 'a', encoding='utf-8') as out:
            out.write(utils.decode_tokens(series))
            print(utils.decode_tokens(series))

  stop_event.set()
  monitor_thread.join()

  avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
  avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
  avg_gpu_usage = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0

  log_resource_usage(start_time, "Decode", "analysis.txt",original_size=None, compressed_size=None, cpu_usage=avg_cpu_usage,
                        memory_usage=avg_memory_usage, gpu_usage=avg_gpu_usage)
  return
 

def encode(rank,world_size,seq_len, temp_dir, compressed_file, FLAGS, series, train_data, last_train_data):
    
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

    if rank == 0:
      start_time = time.time()

    start_index = rank * (FLAGS.batch_size // torch.distributed.get_world_size())
    end_index = (rank + 1) * (FLAGS.batch_size // world_size)
    end_index = min(end_index, len(train_data)) 

    if rank == world_size - 1:
       end_index = min((rank + 1) * (FLAGS.batch_size // world_size), len(train_data))
       """ end_index = min((rank + 1) * (FLAGS.batch_size // world_size), len(train_data)) - seq_len """
    
    

    
    if rank == world_size - 1:
      print(f"[DEBUG] Rank {rank}: start_index={start_index}, end_index={end_index}, len(train_data)={len(train_data)}")
    
    f = [open(os.path.join(temp_dir, compressed_file + '.' + str(i)), 'wb') for i in range(start_index, end_index)]
    bitout = [arithmeticcoding_fast.BitOutputStream(f[i - start_index]) for i in range(start_index, end_index)]
    enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i - start_index]) for i in range(start_index, end_index)]

    print("Encoder initialized")

    torch.distributed.barrier()
    
    prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
    cumul = np.zeros(FLAGS.vocab_size+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)
    
    
    iter_num = len(train_data) // FLAGS.batch_size
    # New iteration depending on the number of GPUs
    iter_num_for_gpu = len(train_data) // (FLAGS.batch_size // world_size)
    print(f"Rank {rank} iter_num_for_gpu: {iter_num_for_gpu} + len(train_data): {len(train_data)}")
    ind = np.array(range(start_index, end_index)) * iter_num

    if np.any(ind >= len(train_data)):
      print(f"[ERROR] RANK: {rank} Out of bounds index found in ind: {ind[ind >= len(train_data)]}")

    ind = np.clip(np.array(range(start_index, end_index)) * iter_num, 0, len(train_data) - 1)

    ind = ind[ind < len(train_data)]
    
    if rank == world_size - 1:
       ind = np.array(range(start_index, end_index)) * iter_num
       iter_num_for_gpu -= 1
    if rank == world_size - 1:
      """ print(f"rank {rank} indsize: {ind.size} iter_num: {iter_num} np.array(range(start_index, end_index)) {np.array(range(start_index, end_index))}") """
    
    for i in range(start_index, end_index):
        for j in range(seq_len):
            enc[i - start_index].write(cumul, series[ind[i - start_index] + j])

    cumul_batch = np.zeros((end_index - start_index, FLAGS.vocab_size + 1), dtype=np.uint64)

    torch.cuda.set_device(rank)
    torch.distributed.barrier()
    model = compress_model.SLiMPerformer(FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,
                                             FLAGS.n_layers, FLAGS.ffn_dim,
                                             FLAGS.n_heads, FLAGS.feature_type, FLAGS.compute_type).cuda()
    
    try:
      model = DDP(model, device_ids=[rank])
      print("Model wrapped in DDP")
    except Exception as e:
      print(f"DDP Initialization Error on rank {rank}: {e}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
    
    print(iter_num_for_gpu)
    dist.barrier()
    for train_index in range(iter_num_for_gpu):     
        model.train()
        try:
            train_batch = train_data[ind, :]
            y = train_batch[:, -1]
        except Exception as e:
            print(f"[ERROR] Failed to retrieve train batch at iteration {train_index}: {e}")
        
        try:
            train_batch = torch.from_numpy(train_batch).cuda().long()
            train_loss, logits = model.module.full_loss(train_batch, with_grad=True)
        except Exception as e:
            print(f"[ERROR] Model forward pass failed at iteration {train_index}: {e}")
        
        try:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except Exception as e:
            print(f"[ERROR] Optimizer step failed at iteration {train_index}: {e}")

        logits = logits.transpose(1, 2)
        prob = F.softmax(logits[:, -1, :], dim=1).detach().cpu().numpy()
        cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

        # Update encoder
        for i in range(start_index, end_index):
            try:
                enc[i - start_index].write(cumul_batch[i - start_index, :], y[i - start_index])
                """ print(f"[DEBUG] Encoder write successful for index {i}") """
            except Exception as e:
                print(f"[ERROR] Encoder write failed for index {i}: {e}")
        
        ind += 1
        ind = np.clip(ind, 0, len(train_data) - 1) # Fix out of bounds error
        # Periodic output
        if train_index % FLAGS.print_step == 0:
            size = 0
            for cf in os.listdir(temp_dir):
                size += os.path.getsize(os.path.join(temp_dir, cf))
            print(f" Iteration on Rank: {rank} : {train_index}: Train loss {train_loss.item() / np.log(2)}, size: {size / (1024 * 1024)} MB")
    
    print(f"[DEBUG] Training completed on rank {rank}")
    dist.barrier()
    # Close encoders and file handles
    for i in range(start_index, end_index):
        enc[i - start_index].finish()
        bitout[i - start_index].close()
        f[i - start_index].close()
    print(f"[DEBUG] Encoders closed on rank {rank}")
    print(f"world_size: {world_size}")
    dist.barrier()
    # Encode the last part of the series
    if rank == world_size - 1:
      if last_train_data is not None:
          print("Last series")
          with open(os.path.join(temp_dir, compressed_file + '.last'), 'wb') as f:
              bitout = arithmeticcoding_fast.BitOutputStream(f)
              enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
              prob = np.ones(FLAGS.vocab_size) / FLAGS.vocab_size
              cumul = np.zeros(FLAGS.vocab_size + 1, dtype=np.uint64)
              cumul[1:] = np.cumsum(prob * 10000000 + 1)

              for j in range(len(last_train_data)):
                  
                  enc.write(cumul, last_train_data[j])
                  
              print("Last encode part don't need inference.")
              enc.finish()
              bitout.close()
    dist.barrier()
    print("Encode finished")

    # Save data for analysis
    if rank == 0:
      input_file_path = FLAGS.input_dir
      original_size = os.path.getsize(input_file_path)
      end_time = time.time()
      
      with open("analysis.txt", 'a', encoding='utf-8') as f:
          elapsed_time = end_time - start_time
          f.write(f"Orginal File Size: {original_size / (1024 * 1024):.2f} MB\n")
          f.write(f"Compression Time: {elapsed_time:.2f} seconds\n")
          f.write("\n")

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

def main(rank, world_size):
  FLAGS(sys.argv)
  init_distributed_mode(rank, world_size)
  print(f"Prozess {rank} verwendet GPU {torch.cuda.current_device()}")
  """ with open("analysis.txt", 'w', encoding='utf-8') as f:
        f.write("Analysis of Compression and Decompression\n") """

  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  main_temp_dir = "{}_{}_{}_{}_bs{}_{}_seq{}_temp".format(
        FLAGS.prefix, FLAGS.vocab_dim, FLAGS.hidden_dim, FLAGS.ffn_dim,
        FLAGS.batch_size, FLAGS.n_layers, FLAGS.seq_len)

  if rank == 0:
        if not os.path.exists(main_temp_dir):
            os.mkdir(main_temp_dir)
        print(f"Hauptordner {main_temp_dir} erstellt.")

  dist.barrier()

  temp_dir = os.path.join(main_temp_dir, f"rank_{rank}_temp")
  if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
  print(f"Rank {rank} erstellt Unterordner {temp_dir}")

  compressed_file = main_temp_dir.replace("_temp", ".compressed")

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
  l = total_length // FLAGS.batch_size * FLAGS.batch_size
  num_batches_per_gpu = l // world_size
  extra_batches = total_length % world_size
  start_idx = rank * num_batches_per_gpu
  end_idx = start_idx + num_batches_per_gpu  #fix out of bounds error

  series_partition = series[start_idx:end_idx]
  
  """ dist.barrier()
  if rank == world_size - 1:
      if total_length % FLAGS.batch_size == 0:
        encode(rank,world_size,FLAGS.seq_len, temp_dir, compressed_file, FLAGS, series_partition, train_data[start_idx:end_idx], None)
      else:
        print(f"REST Rank {rank} processing last data from index {end_idx} to {total_length - 1} train_data[start_idx:l]: {train_data[start_idx:l]}")
        series_partition = series[start_idx:end_idx + FLAGS.seq_len]
        train_data_partition = train_data[start_idx:end_idx]
        encode(rank,world_size,FLAGS.seq_len, temp_dir, compressed_file, FLAGS, series_partition, train_data_partition, series[end_idx:])
  else:
      encode(rank,world_size,FLAGS.seq_len, temp_dir, compressed_file, FLAGS, series_partition, train_data[start_idx:end_idx], None)
 
  
  dist.barrier()

  iterr = (FLAGS.batch_size // world_size) - 1
  rank_counter = 0


  if rank == 0:
      print(f"[DEBUG] Rank 0: Start combining files.")
      print(f"[DEBUG] Initial iterr: {iterr}, Initial rank_counter: {rank_counter}")
      
      # Combined compressed results
      print("[DEBUG] Opening combined file for writing.")
      f = open(compressed_file + '.combined', 'wb')
      
      for i in range(FLAGS.batch_size):
          
          
          # Update rank_counter and iterr
          if i > iterr:
              iterr += FLAGS.batch_size // world_size
              rank_counter += 1
              print(f"[DEBUG] Updated iterr: {iterr}, Updated rank_counter: {rank_counter}")
          
          # Update temp_dir
          temp_dir = os.path.join(main_temp_dir, f"rank_{rank_counter}_temp")
        
          # Load compressed file
          try:
              f_in = open(temp_dir + '/' + compressed_file + '.' + str(i), 'rb')
              byte_str = f_in.read()
              byte_str_len = len(byte_str)
          except FileNotFoundError as e:
              print(f"[ERROR] File not found: {e}")
              continue  # Skip to the next file
          except Exception as e:
              print(f"[ERROR] Failed to read file {temp_dir}/{compressed_file}.{i}: {e}")
              continue

          # Encode and write to the combined file
          try:
              var_int_encode(byte_str_len, f)
              f.write(byte_str)
              f_in.close()
          except Exception as e:
              print(f"[ERROR] Failed to write to combined file: {e}")
      
      # Check 
      if total_length % FLAGS.batch_size != 0:
          temp_dir = os.path.join(main_temp_dir, f"rank_{rank_counter}_temp")
          print(f"[DEBUG] Processing last file in {temp_dir}")
          
          try:
              f_in = open(temp_dir + '/' + compressed_file + '.last', 'rb')
              byte_str = f_in.read()
              byte_str_len = len(byte_str)
              print(f"[DEBUG] Read {byte_str_len} bytes from {temp_dir}/{compressed_file}.last")
              var_int_encode(byte_str_len, f)
              f.write(byte_str)
              f_in.close()
          except FileNotFoundError as e:
              print(f"[ERROR] Last file not found: {e}")
          except Exception as e:
              print(f"[ERROR] Failed to read/write last file: {e}")
      
      f.close()
      print(f"[DEBUG] Closed combined file {compressed_file}.combined")
      
      # Debug: Check the combined file size
      combined_file_size = os.path.getsize(compressed_file + '.combined')
      print(f"Total combined compressed file size: {combined_file_size / (1024 * 1024)} MB")
      with open("analysis.txt", 'a', encoding='utf-8') as f:
        f.write(f"Compressed File Size: {combined_file_size / (1024 * 1024):.2f} MB\n")
        f.write("\n") """

  dist.barrier()

  
  
  #Remove temp file
  print("Start decoding")
  """ shutil.rmtree(temp_dir)   """ 
          
  #Remove all temp files
  shutil.rmtree(main_temp_dir)

  #Now need to create the same dir again
  #Decode
  if rank == 0:             
        if not os.path.exists(main_temp_dir):
            os.mkdir(main_temp_dir)
        print(f"Hauptordner {main_temp_dir} erstellt.")

 
  temp_dir = os.path.join(main_temp_dir, f"rank_{rank}_temp")   

  if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        print(f"Decompression: Rank {rank} erstellt Unterordner {temp_dir}")

  dist.barrier()
  #Split compressed file
  
  f = open(compressed_file+'.combined','rb')
  len_series = len(series) 
  start_index = rank * (FLAGS.batch_size // torch.distributed.get_world_size())
  end_index = ((rank + 1) * (FLAGS.batch_size // torch.distributed.get_world_size()))
  for i in range(start_index, end_index):
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
    print("Decompression: Last part is a full batch.")
    """ decode(rank,temp_dir, compressed_file, FLAGS, len_series, 0) """
  else:
    print("Decompression: Last part is not a full batch.")
    """ last_length = (len_series - FLAGS.seq_len) % FLAGS.batch_size + FLAGS.seq_len
    decode(rank,temp_dir, compressed_file, FLAGS, len_series, last_length) """

  dist.barrier()

  """ if rank == 0:
    combine_decompressed_files(main_temp_dir, world_size, FLAGS.prefix + '.out') """

  dist.destroy_process_group()
  
def combine_decompressed_files(main_temp_dir, num_gpus, output_file):
    with open(output_file, 'wb') as outfile:
        for rank in range(num_gpus):
            input_file = os.path.join(main_temp_dir, f'decompressed_{rank}.out')
            with open(input_file, 'rb') as infile:
                outfile.write(infile.read())

            
            """ os.remove(input_file) """

if __name__ == '__main__':
  world_size = torch.cuda.device_count() 
  mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
  """ app.run(main) """
""" if rank == 0:
    print("RANK 0")
    print(f"RANK 0 Total length: {total_length}")
    print(f"RANK 0 Train Data Size: {train_data.size}")
    print(f"RANK 0 Number of batches per GPU: {num_batches_per_gpu}")
    print(f"RANK 0 Start Index: {start_idx}")
    print(f"RANK 0 End Index: {end_idx}")


  if rank == 1:
    print("RANK 1")
    print(f"RANK 1 Total length: {total_length}")
    print(f"RANK 1 Number of batches per GPU: {num_batches_per_gpu}")
    print(f"RANK 1 Extra batches: {extra_batches}")
    print(f"RANK 1 Start Index: {start_idx}")
    print(f"RANK 1 End Index: {end_idx}")

  if rank == 2:
    print("RANK 2")
    print(f"RANK 2 Total length: {total_length}")
    print(f"RANK 2 Number of batches per GPU: {num_batches_per_gpu}")
    print(f"RANK 2 Extra batches: {extra_batches}")
    print(f"RANK 2 Start Index: {start_idx}")
    print(f"RANK 2 End Index: {end_idx}")
  
  if rank == 3:
    print("RANK 3")
    print(f"RANK 3 Total length: {total_length}")
    print(f"RANK 3 Number of batches per GPU: {num_batches_per_gpu}")
    print(f"RANK 3 Extra batches: {extra_batches}")
    print(f"RANK 3 Start Index: {start_idx}")
    print(f"RANK 3 End Index: {end_idx}")
  
  if rank == 4:
    print("RANK 4")
    print(f"RANK 4 Total length: {total_length}")
    print(f"RANK 4 Number of batches per GPU: {num_batches_per_gpu}")
    print(f"RANK 4 Extra batches: {extra_batches}")
    print(f"RANK 4 Start Index: {start_idx}")
    print(f"RANK 4 End Index: {end_idx}")
  
  if rank == 5:
    print("RANK 5")
    print(f"RANK 5 Total length: {total_length}")
    print(f"RANK 5 Number of batches per GPU: {num_batches_per_gpu}")
    print(f"RANK 5 Extra batches: {extra_batches}")
    print(f"RANK 5 Start Index: {start_idx}")
    print(f"RANK 5 End Index: {end_idx}")
  
  if rank == 6:
    print("RANK 6")
    print(f"RANK 6 Total length: {total_length}")
    print(f"RANK 6 Number of batches per GPU: {num_batches_per_gpu}")
    print(f"RANK 6 Extra batches: {extra_batches}")
    print(f"RANK 6 Start Index: {start_idx}")
    print(f"RANK 6 End Index: {end_idx}")
  
  if rank == 7:
    print("RANK 7")
    print(f"RANK 7 Total length: {total_length}")
    print(f"RANK 7 Number of batches per GPU: {num_batches_per_gpu}")
    print(f"RANK 7 Extra batches: {extra_batches}")
    print(f"RANK 7 Start Index: {start_idx}")
    print(f"RANK 7 End Index: {end_idx}")
    print(f"RANK 7 series_partition: {series_partition}")
    print(f"RANK 7 series_partition ohne FLAGS.seq_len: {series[start_idx:end_idx]}")
    series_partition = series[start_idx:end_idx] """


""" if rank == world_size - 1:
      print(f"RANK 7 ind {ind}")
      print(f"RANK 7 ind[0] {ind[0]}")
      print(f"RANK 7 ind last {ind[-1]}")
      print(f"RANK 7 ind.size {ind.size}")
      print(f"RANK 7 startindex {start_index}")
      print(f"RANK 7 Endindex {end_index}")
      iter_num_for_gpu -= seq_len
      iter_num -= seq_len
    
    if rank == world_size - 2:
      print(f"RANK 6 ind {ind}")
      print(f"RANK 6 ind[0] {ind[0]}")
      print(f"RANK 6 ind last {ind[-1]}")
      print(f"RANK 6 ind.size {ind.size}")
      print(f"RANK 6 startindex {start_index}")
      print(f"RANK 6 Endindex {end_index}") """