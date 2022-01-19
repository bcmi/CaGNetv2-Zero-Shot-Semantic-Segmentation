from __future__ import absolute_import, division, print_function

import os
import sys
import gc
import time
import json
import pickle
import random
import shutil
import operator
import argparse
import numpy as np
import torch
from tools import get_embedding, get_split, get_config, logWritter, MeaninglessError, scores_gzsl, Step_Scheduler, Const_Scheduler
from libs.datasets import get_dataset
#from trainer import Trainer
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets, transforms, utils
backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='***.yaml', help='configuration file for train/val')
    parser.add_argument('--load_model', default='None', help='path to preTrain model')

    parser.add_argument('--first_unseen', type=int, default=13, help='first several pixels are pre-set as unseen categories')
    parser.add_argument('--threshold', type=int, default=17, help='at least threshold pixels in a block should be unseen')
    parser.add_argument('--cc_max', type=int, default=8, help='maximum category count within each block')

    parser.add_argument('--num_iter', type=int, default=10000, help='total iterations of generation')
    parser.add_argument('--start_id', type=int, default=0, help='start index of generation')

    parser.add_argument('--flag', type=float, default=-1, help='split random/pixelcnn++')
    parser.add_argument('--p_base', type=float, default=0, help='p_base / label_count = unseen_adder')
    parser.add_argument('--topk', type=float, default=1, help='topk percentage of categories are chosen for multinomial')
    parser.add_argument('--suffix', default='', help='suffix in name of generated dir')

    return parser.parse_args()


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

fm = 64
class Net(nn.Module):
    def __init__(self, n_class, condition = False):
        super(Net, self).__init__()
        d = 600
        self.conv1 = nn.Sequential(MaskedConv2d('A', d,  fm, 3, 1, 1, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True))
        self.conv2 = nn.Sequential(MaskedConv2d('B', fm, fm, 3, 1, 1, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),)
        self.conv3 = nn.Sequential(MaskedConv2d('B', fm, fm, 3, 1, 1, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),)
        self.conv4 = nn.Sequential(MaskedConv2d('B', fm, fm, 3, 1, 1, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),)
        #self.conv5 = nn.Sequential(MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),)
        #self.conv6 = nn.Sequential(MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),)
        #self.conv7 = nn.Sequential(MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),)
        #self.conv8 = nn.Sequential(MaskedConv2d('B', fm, fm, 7, 1, 3, bias=False), nn.BatchNorm2d(fm), nn.ReLU(True),)
        self.conv9 = nn.Conv2d(fm, n_class, 1)
        self.dropout = nn.Dropout(p=0.5)
        if condition:
            self.b1 = torch.nn.Parameter(torch.Tensor(d,fm))
            self.b2 = torch.nn.Parameter(torch.Tensor(d,fm))
            self.b3 = torch.nn.Parameter(torch.Tensor(d,fm))
            self.b4 = torch.nn.Parameter(torch.Tensor(d,fm))
            #self.b5 = torch.nn.Parameter(torch.Tensor(d,fm))
            #self.b6 = torch.nn.Parameter(torch.Tensor(d,fm))
            #self.b7 = torch.nn.Parameter(torch.Tensor(d,fm))
            #self.b8 = torch.nn.Parameter(torch.Tensor(d,fm))
    def forward(self, x, h=None):
        if h is not None:
            b,_,_,_ = x.size()
            c = 64
            #print (h.size(),self.b1.size())
            #print(torch.matmul(h,self.b1).size())
            x = self.conv1(x) + torch.matmul(h,self.b1).permute(0,3,1,2).contiguous()#.view(b,c,1,1)
            x = self.conv2(x) + torch.matmul(h,self.b2).permute(0,3,1,2).contiguous()#.view(b,c,1,1)
            x = self.dropout(self.conv3(x) + torch.matmul(h,self.b3).permute(0,3,1,2).contiguous())#.view(b,c,1,1)
            x = self.dropout(self.conv4(x) + torch.matmul(h,self.b4).permute(0,3,1,2).contiguous())#.view(b,c,1,1)
            #x = self.conv5(x) + torch.matmul(h,self.b5).permute(0,3,1,2).contiguous()#.view(b,c,1,1)
            #x = self.conv6(x) + torch.matmul(h,self.b6).permute(0,3,1,2).contiguous()#.view(b,c,1,1)
            #x = self.dropout(self.conv7(x) + torch.matmul(h,self.b7).permute(0,3,1,2).contiguous())#.view(b,c,1,1)
            #x = self.dropout(self.conv8(x) + torch.matmul(h,self.b8).permute(0,3,1,2).contiguous())#.view(b,c,1,1)
            return self.conv9(x)
        else:
            return self.conv9((self.conv4(self.conv3(self.conv2(self.conv1(x))))))


args = parse_args()
config = get_config(args.config)

vals_cls, valu_cls, all_labels, visible_classes, visible_classes_test, train, val, sampler, visibility_mask, cls_map, cls_map_test = get_split(config)

class_emb = get_embedding(config)
class_emb_vis = class_emb[visible_classes]
class_emb_vis_ = torch.zeros((config['ignore_index'] + 1 - class_emb_vis.shape[0], class_emb_vis.shape[1]), dtype = torch.float32)
class_emb_vis_aug = torch.cat((class_emb_vis, class_emb_vis_), dim=0).cuda()
class_emb_all = class_emb[visible_classes_test].cuda()
print (class_emb_all.size())

Em = nn.Embedding.from_pretrained(class_emb_all)
Em = Em.cuda()
Em.weight.requires_grad = False

unseen_range = (vals_cls.size, vals_cls.size+valu_cls.size-1) # (15, 19) for voc12
print('Unseen index ranging from {0} to {1}'.format(unseen_range[0], unseen_range[1]))
label_count = config['dis']['out_dim_cls'] - 1 # 20 for voc12
print('Label count equals {0}'.format(label_count))
assert(unseen_range[1]+1 == label_count)
assert(class_emb_all.size()[0] == label_count)
assert(config['num_unseen'] == unseen_range[1]-unseen_range[0]+1)

net = Net(label_count, True)
net.cuda()
net = torch.nn.DataParallel(net)

assert(os.path.exists(args.load_model))
state_dict = torch.load(args.load_model)['state_dict']
net.load_state_dict(state_dict, strict=True)
print("Pre-train model {0} successfully loaded".format(args.load_model))

torch.backends.cudnn.enabled = True
net.train(False)

cc_max = args.cc_max # 3 for default
threshold = args.threshold  # 6 # at least 'threshold' pixels in a block should be unseen
unseen_adder = args.p_base / label_count # adder gets smaller when label_count increases / p_base decreases
print('threshold = {0}, unseen_adder = {1:.3f}'.format(threshold, unseen_adder))

assert(args.topk > 0.0 and args.topk <= 1.0)
label_top = int(label_count * args.topk)
print('label_top equals {0}'.format(label_top))
assert(label_top > 0)

bs = 6400 #16384
block_size = 5
assert(args.first_unseen>=0 and args.first_unseen<=block_size**2)
datatype = np.int64

res = torch.zeros((0, block_size, block_size)).long().cuda()
p_unseen = torch.zeros((bs, label_count)).cuda()
p_unseen[:, unseen_range[0]:unseen_range[1]+1] += unseen_adder
margain = (torch.LongTensor(range(bs)) * label_top).long().cuda() # shape = (bs)

prior = np.ones((config['dis']['out_dim_cls']-1))
for k in range(config['dis']['out_dim_cls']-config['num_unseen']-1,config['dis']['out_dim_cls']-1):
    prior[k] = prior[k]+config['gen_unseen_rate']
prior_ = prior/np.linalg.norm(prior, ord=1)

num_iter = args.num_iter
display_iter = 100
count_random, count_pixelcnn = 0, 0
print('Start generation ...')

for t in range(num_iter):

    h0 = (unseen_range[1] - unseen_range[0] + 1) * torch.rand((bs)) + unseen_range[0]
    #h0 = h0.unsqueeze(1).unsqueeze(1).repeat(1, block_size, block_size).long().cuda()
    h0 = h0.unsqueeze(1).repeat(1, block_size**2).long().cuda()
    h0[h0==unseen_range[1]+1] = unseen_range[1]
    assert (h0.max() <= unseen_range[1] and h0.min() >= unseen_range[0])
    h1 = h0.view(bs, block_size, block_size)
    h = Em(h1).float().contiguous()

    if np.random.rand() > args.flag:
        # sample = torch.LongTensor(bs, block_size, block_size).cuda()
        sample = torch.LongTensor(bs, block_size**2).cuda()
        sample.fill_(0)
        if args.first_unseen != 0:
            sample[:,:args.first_unseen] = h0[:,:args.first_unseen]
        sample = sample.view(bs, block_size, block_size)

        for ij in range(args.first_unseen, block_size**2):
                i = ij // block_size
                j = ij % block_size

        # for i in range(block_size):
        #     for j in range(block_size):

                input = Em(sample).permute(0,3,1,2).contiguous()
                out = net(input, h)
                probs = F.softmax(out[:, :, i, j], dim=1).data
                probs += p_unseen

                #probs = F.softmax(probs, dim=1)
                #sample[:, i, j] = torch.multinomial(probs, 1).long().squeeze(1)

                probs_k, chosen_labels = torch.topk(probs, label_top, dim=1) # shapes = (bs, label_top), chosen_labels < label_count
                probs_k = F.softmax(probs_k, dim=1)
                selected_indices = torch.multinomial(probs_k, 1).long().squeeze(1) # shape = (bs)
                sample[:, i, j] = chosen_labels.view(-1)[(selected_indices + margain).long()].long() # shape = (bs)

        #utils.save_image(sample_numpy, 'blocks/sample_{:02d}_{:02d}.png'.format(epoch,k), nrow=12, padding=0)
        unseen_ones = torch.where(sample>=unseen_range[0], torch.ones(sample.shape).long().cuda(), torch.zeros(sample.shape).long().cuda()).long().view(-1, block_size**2)
        gen_sum = torch.sum(unseen_ones, dim=1).long()
        unseen_exist_index = torch.where(gen_sum>=threshold)[0].long()
        temp_unseen_0 = sample[unseen_exist_index]

        # add category count restriction here
        gt_sample = temp_unseen_0.view(-1,block_size**2)
        ones = torch.ones(gt_sample.shape).long().cuda()
        zeros = torch.zeros(gt_sample.shape).long().cuda()
        one = torch.ones(gt_sample.shape[0]).long().cuda()
        zero = torch.zeros(gt_sample.shape[0]).long().cuda()
        cal = torch.zeros(gt_sample.shape[0]).long().cuda()
        for cate in range(label_count):
            a1 = torch.where(gt_sample==cate, ones, zeros).long()
            a2 = torch.sum(a1, dim=1).long()
            a3 = torch.where(a2!=0, one, zero).long()
            cal += a3
        cc_lt_max_index = torch.where(cal<=cc_max)[0].long()
        temp_unseen = temp_unseen_0[cc_lt_max_index]

        count_pixelcnn += temp_unseen.shape[0]

    else:
        sample = torch.LongTensor(np.random.choice(a=range(label_count), size=(bs, 1, 1), replace=True, p=prior_)).cuda()
        sample = sample.repeat(1, block_size, block_size)
        temp_unseen = sample
        count_random += temp_unseen.shape[0]

    res = torch.cat((res, temp_unseen), dim=0)

    if (t+1) % display_iter == 0:
        print('[{0}/{1}], count_pixelcnn = {2}, count_random = {3}'.format(t+1, num_iter, count_pixelcnn, count_random))


res = res.cpu().numpy().astype(datatype)

res_index = list(range(res.shape[0]))
for i in range(3):
    random.shuffle(res_index)
res_index = np.array(res_index, dtype=datatype)
res = res[res_index]

bs_new = 1600 #4096
times = res.shape[0] // bs_new
print("OutputFile total: {0}".format(times))

#outputdir = './gen_{0}_{1}_{2:.3f}_{3:.3f}_{4}_{5:.3f}_{6}{7}'.format(config['dataset'], threshold, unseen_adder, args.flag, args.cc_max, args.topk, args.first_unseen, args.suffix)
outputdir = './gen_{0}_pa{1}_fu{2}_th{3}_cc{4}{5}'.format(config['dataset'], block_size, args.first_unseen, threshold, args.cc_max, args.suffix)
print('Outputdir: ' + outputdir)
if not os.path.exists(outputdir):
    os.mkdir(outputdir)

start_id = args.start_id
for i in range(times):
    s, t = i*bs_new, (i+1)*bs_new
    temp_res = res[s:t,:,:]  # (bs_new, block_size, block_size)
    filename = os.path.join(outputdir, str(start_id+i) + '.npy')
    np.save(filename, temp_res)
