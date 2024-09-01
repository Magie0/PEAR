import collections
import gzip
import os
import time
import utils
import struct
from absl import app
from absl import flags
from absl import logging
import shutil
import pandas as pd

import numpy as np
import torch
import torch.nn.functional as F

import PEARmodel as compress_model
import arithmeticcoding_fast3
import arithmeticcoding_fast2
import utils

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
torch.set_printoptions(profile="full") 
FLAGS = flags.FLAGS

# Model parameters
flags.DEFINE_integer('batch_size', 512, 'Batch size for training.')
flags.DEFINE_float('learning_rate', 1e-3, 'Adam Optimizer learning rate.')
flags.DEFINE_integer('hidden_dim', 256, 'Feature dimension.')
flags.DEFINE_integer('vocab_dim', 16, 'Feature dimension.')
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
flags.DEFINE_string('gpu_id', '1', 'ID of GPU.')
flags.DEFINE_integer('random_seed', 0, 'Random seed for both Numpy and Torch.')
flags.DEFINE_integer('print_step', 1000, 'Interval to print metrics.')
# Dataset parameters
flags.DEFINE_integer('seq_len', 1, 'Maximum sequence length (L).')
flags.DEFINE_integer('vocab_size', 256, 'Vocabulary size of data.')
flags.DEFINE_string('input_dir', 'aaa', 'input data dir')
flags.DEFINE_string('prefix', 'text8', 'output dir')


def decode(temp_dir, compressed_file, FLAGS, len_series, last):
  
  bs = FLAGS.batch_size

  iter_num = (len_series - FLAGS.seq_len) // FLAGS.batch_size
  
  ind = np.array(range(bs))*iter_num
  print(iter_num - FLAGS.seq_len)
  series_2d = np.zeros((bs,iter_num), dtype = np.uint8).astype('int')

  f = [open(temp_dir+"/"+compressed_file+'.'+str(i),'rb') for i in range(bs)]
  bitin = [arithmeticcoding_fast3.BitInputStream1(f[i]) for i in range(bs)]
  dec = [arithmeticcoding_fast3.ArithmeticDecoder(32, bitin[i]) for i in range(bs)]

  prob = np.ones(FLAGS.vocab_size)/FLAGS.vocab_size
  cumul = np.zeros(FLAGS.vocab_size+1, dtype = np.uint64)
  cumul[1:] = np.cumsum(prob*10000000 + 1)

  # Decode first K symbols in each stream with uniform probabilities
  print(min(FLAGS.seq_len, iter_num))
  for i in range(bs):
    for j in range(min(FLAGS.seq_len, iter_num)):
      series_2d[i,j] = dec[i].read(cumul, FLAGS.vocab_size)
  print("shape:",series_2d.shape)
  
  cumul_batch = np.zeros((bs, FLAGS.vocab_size+1), dtype = np.uint64)

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  model = compress_model.mlpCompressor(FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,
                                             FLAGS.n_layers, FLAGS.ffn_dim,
                                             FLAGS.n_heads, FLAGS.batch_size).cuda()
  
  print(model)

  #optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay, betas=(.9, .999))
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-7)
  training_start = time.time()
  for train_index in range(iter_num-FLAGS.seq_len):
    model.train()
    train_batch = torch.LongTensor(series_2d[:, train_index:train_index + FLAGS.seq_len]).cuda()
    logits, beta = model.forward(train_batch)
    prob = logits[:, -1, :]
    prob = F.softmax(prob, dim=1).detach().cpu().numpy()
    cumul_batch[:,1:] = np.cumsum(prob*10000000 + 1, axis = 1)

    # Decode with Arithmetic Encoder
    for i in range(bs):
      series_2d[i,train_index+FLAGS.seq_len] = dec[i].read(cumul_batch[i,:], FLAGS.vocab_size)
    #print(series_2d.shape)
    
    logits = logits.transpose(1, 2)
    label = torch.from_numpy(series_2d[:, train_index+1:train_index+FLAGS.seq_len+1]).cuda()
    train_loss = torch.nn.functional.cross_entropy(logits[:, :, -1], label[:, -1], reduction='mean')
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    if (train_index+1) % FLAGS.print_step == 0:
      print(train_index, ":", train_loss.item()/np.log(2))
  
    
  out = open('decompressed_out', 'w')
  for i in range(len(series_2d)):
    out.write(utils.decode_tokens(series_2d[i]))
  
  
  for i in range(bs):
    bitin[i].close()
    f[i].close()

  if last:
    series = np.zeros(last, dtype = np.uint8).astype('int')
    f = open(temp_dir+"/"+compressed_file+'.last','rb')
    bitin = arithmeticcoding_fast3.BitInputStream1(f)
    dec = arithmeticcoding_fast3.ArithmeticDecoder(32, bitin)
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
    return

from multiprocessing import Manager
import multiprocessing
import logging
import mmap
import threading
import time

num_processes = 16
seed = 611

def copy_to_shared_memory(shared_cumul, shared_y, cumul_batch, y, start_index, end_index):
    np.copyto(np.frombuffer(shared_cumul.get_obj(), dtype=np.uint64).reshape((-1,))[
              start_index * (cumul_batch.shape[1]): end_index * (cumul_batch.shape[1])],
              cumul_batch[start_index:end_index].flatten())
    np.copyto(np.frombuffer(shared_y.get_obj(), dtype=np.int32)[start_index:end_index], y[start_index:end_index])

def process_task(start_index, end_index, shared_cumul, shared_y, shutdown_flag, temp_dir, compressed_file, task_id, bs, vocab_size, barrier):
    try:
        f = [open(temp_dir + "/" + compressed_file + '.' + str(i), 'wb') for i in range(start_index, end_index)]
        bitout = [arithmeticcoding_fast3.BitOutputStream1(f[i - start_index]) for i in range(start_index, end_index)]
        enc = [arithmeticcoding_fast3.ArithmeticEncoder(32, bitout[i - start_index]) for i in range(start_index, end_index)]

        while True:
            if shutdown_flag.value:
                break

            barrier.wait()  # 等待主进程更新数据

            if shutdown_flag.value:
                break

            cumul_batch = np.frombuffer(shared_cumul.get_obj(), dtype=np.uint64).reshape((bs, vocab_size + 1))
            y = np.frombuffer(shared_y.get_obj(), dtype=np.int32)

            for i in range(start_index, end_index):
                enc[i - start_index].write(cumul_batch[i, :], y[i])

            barrier.wait()  # 通知主进程任务完成

    except Exception as e:
        print(f"任务 {task_id} 遇到错误: {e}")
        raise
    finally:
        for i in range(end_index - start_index):
            enc[i].finish()
            bitout[i].close()
            f[i].close()

def data_handling(shared_cumul, shared_y, logits, y, num_processes, chunk_size, bs, FLAGS, queue):
    logits = logits.squeeze(2)
    prob = F.softmax(logits, dim=1).detach().cpu().numpy()  
    cumul_batch = np.zeros((bs, FLAGS.vocab_size + 1), dtype=np.uint64)
    cumul_batch[:, 1:] = np.cumsum(prob * 10000000 + 1, axis=1)

    copy_to_shared_memory(shared_cumul, shared_y, cumul_batch, y, 0, bs)
    queue.put(1)

def model_prediction_task(shared_cumul, shared_y, queue, shutdown_flag, FLAGS, series, train_data, chunk_size, num_processes):
    model_start_time = time.time()
    bs = FLAGS.batch_size
    iter_num = len(train_data) // FLAGS.batch_size
    ind = np.array(range(bs)) * iter_num
    iter_num -= FLAGS.seq_len

    model = compress_model.mlpCompressor(FLAGS.vocab_size, FLAGS.vocab_dim, FLAGS.hidden_dim,
                                         FLAGS.n_layers, FLAGS.ffn_dim,
                                         FLAGS.n_heads, FLAGS.batch_size).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=1e-7)

    print(model)
    print(iter_num)
    for train_index in range(iter_num):
        if shutdown_flag.value:
            break

        model.train()
        train_batch = train_data[ind, :]
        y = train_batch[:, -1]

        train_batch = torch.from_numpy(train_batch).cuda().long()
        logits, emb = model.forward(train_batch[:, :-1])
        logits = logits.transpose(1, 2)
        train_loss = torch.nn.functional.cross_entropy(
                logits[:, :, -1], train_batch[:, -1], reduction='mean')
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # 使用独立线程处理 logits 后处理和数据拷贝操作
        data_thread = threading.Thread(target=data_handling, args=(shared_cumul, shared_y, logits, y, num_processes, chunk_size, bs, FLAGS,queue))
        data_thread.start()

        ind += 1
        #如果是最后一轮，需要等待所有线程处理完毕
        if train_index == iter_num - 1:
            data_thread.join() 
        if (train_index + 1) % FLAGS.print_step == 0:
            print(train_index, ":", train_loss.item() / np.log(2), "Duration:", (time.time() - model_start_time)/3600, "hours")

    model_end_time = time.time()
    print(f"模型预测时间: {(model_end_time - model_start_time)/3600} h")
    queue.put('exit')  # 通知主进程模型预测已完成
    print("模型预测进程退出")

def encode(temp_dir, compressed_file, FLAGS, series, train_data, last_train_data):
    start_time = time.time()
    bs = FLAGS.batch_size
    chunk_size = bs // num_processes 

    prob = np.ones(FLAGS.vocab_size) / FLAGS.vocab_size
    cumul = np.zeros((bs, FLAGS.vocab_size + 1), dtype=np.uint64)
    cumul[:, 1:] = np.cumsum(prob * 10000000 + 1)
    iter_num = len(train_data) // FLAGS.batch_size
    ind = np.array(range(bs)) * iter_num

    shared_cumul = multiprocessing.Array(np.ctypeslib.as_ctypes_type(np.uint64), bs * (FLAGS.vocab_size + 1))
    shared_y = multiprocessing.Array(np.ctypeslib.as_ctypes_type(np.int32), bs)
    shutdown_flag = multiprocessing.Value('b', False)
    queue = multiprocessing.Queue()
    barrier = multiprocessing.Barrier(num_processes + 1)

    # 启动编码进程
    processes = []
    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i != num_processes - 1 else bs
        p = multiprocessing.Process(target=process_task, args=(start_index, end_index, shared_cumul, shared_y, shutdown_flag, temp_dir, compressed_file, i, bs, FLAGS.vocab_size, barrier))
        processes.append(p)
        p.start()
    
        y = np.zeros((FLAGS.seq_len, bs), dtype=series.dtype)

    for i in range(bs):
        for j in range(FLAGS.seq_len):
            y[j][i] = series[ind[i] + j]

    for j in range(FLAGS.seq_len):
        copy_start_time = time.time()
        threads = []
        for i in range(num_processes):
            start_index = i * chunk_size
            end_index = (i + 1) * chunk_size if i != num_processes - 1 else bs
            t = threading.Thread(target=copy_to_shared_memory, args=(shared_cumul, shared_y, cumul, y[j], start_index, end_index))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        barrier.wait()  # 通知子进程数据已更新
        barrier.wait()  # 等待子进程完成当前任务

    # 启动模型预测进程
    model_process = multiprocessing.Process(target=model_prediction_task, args=(shared_cumul, shared_y, queue, shutdown_flag, FLAGS, series, train_data, chunk_size, num_processes))
    model_process.start()

    # 主进程负责管理数据传递
    cycle_start_time = time.time()
    while True:
        signal = queue.get()
        if signal == 'exit':
           break
        barrier.wait()  # 通知编码进程数据已更新
        barrier.wait()  # 等待编码进程完成当前任务
    cycle_end_time = time.time()
    print(f"循环时间: {(cycle_end_time - cycle_start_time)/3600} hour")

    model_process.join()
    time.sleep(0.1)
    shutdown_flag.value = True
    barrier.wait()  # 通知编码进程退出
    for p in processes:
        p.join()

    total_duration = time.time() - start_time
    print(f"总压缩时间: {total_duration / 3600} 小时")

    if last_train_data is not None:
        print("last series")
        f = open(temp_dir + "/" + compressed_file + '.last', 'wb')
        bitout = arithmeticcoding_fast2.BitOutputStream1(f)
        enc = arithmeticcoding_fast2.ArithmeticEncoder(32, bitout)
        prob = np.ones(FLAGS.vocab_size) / FLAGS.vocab_size
        cumul = np.zeros(FLAGS.vocab_size + 1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob * 10000000 + 1)

        for j in range(len(last_train_data)):
            enc.write(cumul, last_train_data[j])
        print("Last encode part don't need inference.")

        enc.finish()
        bitout.close()
        f.close()
    size = 0
    for cf in os.listdir(temp_dir):
        size += os.path.getsize(temp_dir + "/" + cf)
    print(size / (1024 * 1024))
    total_duration = time.time() - start_time
    print(f"总压缩时间2: {total_duration / 3600} 小时")

    
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

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_id
  np.random.seed(FLAGS.random_seed)
  torch.manual_seed(FLAGS.random_seed)

  temp_dir = "{}_{}_{}_{}_bs{}_{}_seq{}_temp".format(FLAGS.prefix, FLAGS.vocab_dim, FLAGS.hidden_dim, FLAGS.ffn_dim, FLAGS.batch_size, FLAGS.n_layers, FLAGS.seq_len)
  compressed_file = temp_dir.replace("_temp", ".compressed")
  
  if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
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