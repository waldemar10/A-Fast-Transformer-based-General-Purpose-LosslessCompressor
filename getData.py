""" import os
import time
import threading
import psutil
import GPUtil
from compressor import main as compressor_main
from absl import flags, app
import sys

sys.stdout.reconfigure(encoding='utf-8')

FLAGS = flags.FLAGS

def get_file_size(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def calculate_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

def log_results(output_file, original_size, compressed_size, compression_ratio, compression_time, decompression_time, avg_cpu_usage, avg_memory_usage, avg_gpu_usage):
    with open(output_file, 'w', encoding='utf-8') as f: 
        f.write(f"Original File Size: {original_size / (1024*1024):.2f} MB\n")
        f.write(f"Compressed File Size: {compressed_size / (1024*1024):.2f} MB\n")
        f.write(f"Compression Ratio: {compression_ratio:.2f}\n")
        f.write(f"Compression Time: {compression_time:.2f} seconds\n")
        f.write(f"Decompression Time: {decompression_time:.2f} seconds\n")
        f.write(f"Average CPU Usage: {avg_cpu_usage:.2f}%\n")
        f.write(f"Average Memory Usage: {avg_memory_usage:.2f} MB\n")
        f.write(f"Average GPU Usage: {avg_gpu_usage:.2f}%\n")

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

def main(_):
    try:
        original_file = FLAGS.input_dir
        compressed_file = "{}_{}_{}_{}_bs{}_{}_seq{}.compressed.combined".format(
            FLAGS.prefix, FLAGS.vocab_dim, FLAGS.hidden_dim, FLAGS.ffn_dim,
            FLAGS.batch_size, FLAGS.n_layers, FLAGS.seq_len)
        
        original_size = get_file_size(original_file)

        cpu_usages, memory_usages, gpu_usages = [], [], []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=monitor_resources, args=(cpu_usages, memory_usages, gpu_usages, stop_event))
        monitor_thread.start()
        
        start_time = time.time()
        compressor_main(None)  # Komprimierung
        compression_time = time.time() - start_time

        compressed_size = get_file_size(compressed_file)
        compression_ratio = calculate_compression_ratio(original_size, compressed_size)

        start_time = time.time()
        compressor_main(None)  # Dekomprimierung
        decompression_time = time.time() - start_time

        stop_event.set()
        monitor_thread.join()

        avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
        avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
        avg_gpu_usage = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0

        output_file = "compression_results.txt"
        log_results(output_file, original_size, compressed_size, compression_ratio, compression_time, decompression_time, avg_cpu_usage, avg_memory_usage, avg_gpu_usage)

        print(f"Results saved to {output_file}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(main)
 """