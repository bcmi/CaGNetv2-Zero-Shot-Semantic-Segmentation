import os
import shutil
import random
import argparse
import torch
import numpy as np
from tools import get_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='***.yaml', help='configuration file for train/val')
parser.add_argument('--inputdir', default='./xxx', help='input dir to generation')
parser.add_argument('--fuse_ratio', type=int, default=8, help='fuse ratio of pure blocks to pixelcnn blocks')
parser.add_argument('--start_id', type=int, default=0, help='start index of generation')
args = parser.parse_args()
config = get_config(args.config)

fuse_ratio = args.fuse_ratio
bs = 4096
block_size = 3
datatype = np.int64

inputdir = args.inputdir
assert(os.path.exists(inputdir))
fall = os.listdir(inputdir)
ftotal = len(fall)

outputdir = '{0}_fuse{1}'.format(inputdir, fuse_ratio)
assert(not os.path.exists(outputdir))
os.mkdir(outputdir)


res = np.zeros((0, block_size, block_size), dtype=datatype)

for f in fall:
    fir = os.path.join(inputdir, f)
    gen_block = np.load(fir)
    assert(gen_block.shape == (bs, block_size, block_size))
    res = np.concatenate((res, gen_block), axis=0)

res_index = list(range(res.shape[0]))
for i in range(3):
    random.shuffle(res_index)
res_index = np.array(res_index, dtype=datatype)
res = res[res_index]
assert(res.shape[0] == bs * ftotal)


pure_count = bs * ftotal * fuse_ratio
label_count = config['dis']['out_dim_cls'] - 1 # 20 for voc12
print('Label count: {0}'.format(label_count))
prior = np.ones((config['dis']['out_dim_cls']-1))
for k in range(config['dis']['out_dim_cls']-config['num_unseen']-1,config['dis']['out_dim_cls']-1):
    prior[k] = prior[k]+config['gen_unseen_rate']
prior_ = prior/np.linalg.norm(prior, ord=1)
sample = torch.LongTensor(np.random.choice(a=range(label_count), size=(pure_count, 1, 1), replace=True, p=prior_))
sample = sample.repeat(1, block_size, block_size)
pure_block_all = sample.numpy().astype(datatype)

res_fuse = np.concatenate((res, pure_block_all), axis=0)

res_fuse_index = list(range(res_fuse.shape[0]))
for i in range(3):
    random.shuffle(res_fuse_index)
res_fuse_index = np.array(res_fuse_index, dtype=datatype)
res_fuse = res_fuse[res_fuse_index]
assert(res_fuse.shape[0] == bs * ftotal * (fuse_ratio+1))

times = res_fuse.shape[0] // bs
assert(times == ftotal * (fuse_ratio+1))
print("OutputFile total: {0}".format(times))
start_id = args.start_id
for i in range(times):
    s, t = i*bs, (i+1)*bs
    temp_res = res_fuse[s:t,:,:]  # (bs, block_size, block_size)
    filename = os.path.join(outputdir, str(start_id+i) + '.npy')
    np.save(filename, temp_res)
