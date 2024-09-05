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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
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
flags.DEFINE_integer('random_seed', 0, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('print_step', 1000, 'Interval to print metrics.')
# Dataset parameters
flags.DEFINE_integer('seq_len', 8, 'Maximum sequence length (L).')
flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size of data.')
flags.DEFINE_string('input_dir', 'aaa', 'input data dir')
flags.DEFINE_string('prefix', 'text8', 'output dir')

def setup_distributed(rank, world_size):
    # Initialisiere das Prozess-Group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
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

def decode(rank, world_size, temp_dir, compressed_file, FLAGS, len_series, last):
    setup_distributed(rank, world_size)

    cpu_usages, memory_usages, gpu_usages = [], [], []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(cpu_usages, memory_usages, gpu_usages, stop_event))
    monitor_thread.start()

    start_time = time.time()
    bs = FLAGS.batch_size

    iter_num = (len_series - FLAGS.seq_len) // FLAGS.batch_size
    ind = np.array(range(bs)) * iter_num

    print(iter_num - FLAGS.seq_len)
    series_2d = np.zeros((bs, iter_num), dtype=np.uint8).astype('int')

    f = [open(temp_dir+"/"+compressed_file+'.'+str(i), 'rb') for i in range(bs)]
    bitin = [arithmeticcoding_fast.BitInputStream(f[i]) for i in range(bs)]
    dec = [arithmeticcoding_fast.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

    prob = np.zeros((FLAGS.vocab_size + 1, FLAGS.vocab_size + 1), dtype=np.uint64)
    prob[0, :] = np.zeros((FLAGS.vocab_size + 1), dtype=np.uint64)
    prob[1, :] = np.ones((FLAGS.vocab_size + 1), dtype=np.uint64)

    for i in range(bs):
        for j in range(FLAGS.vocab_size):
            prob[j, :] = (j * 10000000 + 1)

    for i in range(iter_num - FLAGS.seq_len):
        model = compress_model.SLiMPerformer(
            FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,
            FLAGS.n_layers, FLAGS.ffn_dim, FLAGS.n_heads,
            FLAGS.feature_type, FLAGS.compute_type
        ).to(rank)

        model = DDP(model, device_ids=[rank])
        model.load_state_dict(torch.load(f"{temp_dir}/model.pth"))

        optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(0.9, 0.999))

        for j in range(bs):
            series_2d[j, :] = utils.decode_tokens(dec[j].decode()[:(len_series // bs)])

    stop_event.set()
    monitor_thread.join()

    avg_cpu_usage = np.mean(cpu_usages)
    avg_memory_usage = np.mean(memory_usages)
    avg_gpu_usage = np.mean(gpu_usages)

    log_resource_usage(start_time, "Decoding", "decoding_log.txt", cpu_usage=avg_cpu_usage,
                       memory_usage=avg_memory_usage, gpu_usage=avg_gpu_usage)

    cleanup_distributed()

def encode(rank, world_size, temp_dir, input_file, FLAGS):
    setup_distributed(rank, world_size)

    cpu_usages, memory_usages, gpu_usages = [], [], []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_resources, args=(cpu_usages, memory_usages, gpu_usages, stop_event))
    monitor_thread.start()

    start_time = time.time()
    bs = FLAGS.batch_size

    # Prepare the data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    tokens = utils.encode_tokens(data)
    len_tokens = len(tokens)
    
    series_2d = np.zeros((bs, len_tokens // bs), dtype=np.uint8)
    for i in range(bs):
        series_2d[i, :] = tokens[i * (len_tokens // bs):(i + 1) * (len_tokens // bs)]
    
    cumul = np.zeros(FLAGS.vocab_size + 1, dtype=np.uint64)
    prob = np.ones(FLAGS.vocab_size) / FLAGS.vocab_size
    cumul[1:] = np.cumsum(prob * 10000000 + 1)

    # Model setup
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)

    model = compress_model.SLiMPerformer(
        FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,
        FLAGS.n_layers, FLAGS.ffn_dim, FLAGS.n_heads,
        FLAGS.feature_type, FLAGS.compute_type
    ).to(rank)

    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(0.9, 0.999))

    for i in range(0, len_tokens, bs):
        model.train()
        batch = torch.LongTensor(series_2d[:, i:i + bs]).to(rank)
        logits = model(batch)
        prob = F.softmax(logits[:, -1, :], dim=1).detach().cpu().numpy()

        # Write encoded data
        bitout = [arithmeticcoding_fast.BitOutputStream(open(f"{temp_dir}/compressed_file.{i}", 'wb')) for i in range(bs)]
        enc = [arithmeticcoding_fast.ArithmeticEncoder(32, bitout[i]) for i in range(bs)]

        cumul_batch = np.zeros((bs, FLAGS.vocab_size + 1), dtype=np.uint64)
        cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

        for i in range(bs):
            for j in range(batch.shape[1]):
                enc[i].write(cumul_batch[i, :], batch[i, j])
        
        for i in range(bs):
            bitout[i].close()

        optimizer.zero_grad()

    # Write final compressed file
    with open(f"{temp_dir}/compressed_file.last", 'wb') as f:
        bitout = arithmeticcoding_fast.BitOutputStream(f)
        enc = arithmeticcoding_fast.ArithmeticEncoder(32, bitout)
        for i in range(len_tokens % bs):
            enc.write(cumul_batch[i, :], series_2d[i, -1])
        bitout.close()

    stop_event.set()
    monitor_thread.join()

    avg_cpu_usage = np.mean(cpu_usages)
    avg_memory_usage = np.mean(memory_usages)
    avg_gpu_usage = np.mean(gpu_usages)

    log_resource_usage(start_time, "Encoding", "encoding_log.txt", original_size=len(tokens), cpu_usage=avg_cpu_usage,
                       memory_usage=avg_memory_usage, gpu_usage=avg_gpu_usage)

    cleanup_distributed()
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

def main(argv):
    del argv  # Unused
    compressed_file = FLAGS.prefix
    temp_dir = FLAGS.input_dir
    batch_size = FLAGS.batch_size
    hidden_dim = FLAGS.hidden_dim
    vocab_dim = FLAGS.vocab_dim
    seq_len = FLAGS.seq_len
    learning_rate = FLAGS.learning_rate
    ffn_dim = FLAGS.ffn_dim

    len_series = 1000000  # Dummy value, set according to your dataset size
    last = False  # Set according to your needs

    world_size = torch.cuda.device_count()
    rank = 0  # Assuming single GPU for simplicity. Adjust if using multiple GPUs

    encode(rank, world_size, temp_dir, compressed_file, FLAGS)
    decode(rank, world_size, temp_dir, compressed_file, FLAGS, len_series, last)

if __name__ == "__main__":
    app.run(main)