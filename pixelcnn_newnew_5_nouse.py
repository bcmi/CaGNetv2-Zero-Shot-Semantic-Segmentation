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
import numpy.ma as ma
import torch
from tools import get_embedding, get_split, get_config, logWritter, MeaninglessError, scores_gzsl, Step_Scheduler, Const_Scheduler
from libs.datasets import get_dataset
from trainer import Trainer
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
    parser.add_argument('--resume_from', type=int, default=0, help='start epoch index')
    parser.add_argument('--val', action='store_true', default=False, help='only do validation if set True')

    return parser.parse_args()

def fill_in_hole(a, fill_value=255, sh=(-1,1), ax=(1,2)):
    b = ma.masked_array(a, a==fill_value)
    for shift in sh:
        for axis in ax:        
            b_shifted = np.roll(b, shift=shift, axis=axis)
            idx = ~b_shifted.mask * b.mask
            b[idx] = b_shifted[idx]
    return b.data, b.mask


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
logger = logWritter('blocks_new_5/log_{0}.txt'.format(config['dataset']))

vals_cls, valu_cls, all_labels, visible_classes, visible_classes_test, train, val, sampler, visibility_mask, cls_map, cls_map_test = get_split(config)

dataset = get_dataset(config['DATAMODE'])(
        train=train, 
        test=None,
        root=config['ROOT'],
        split=config['SPLIT']['TRAIN'],
        base_size=513,
        crop_size=config['IMAGE']['SIZE']['TRAIN'],
        mean=(config['IMAGE']['MEAN']['B'], config['IMAGE']['MEAN']['G'], config['IMAGE']['MEAN']['R']),
        warp=config['WARP_IMAGE'],
        scale=(0.5, 1.5),
        flip=True,
        visibility_mask=visibility_mask
    )

tr = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=6,
        num_workers=config['NUM_WORKERS'],
        sampler=sampler
    )

dataset_test = get_dataset(config['DATAMODE'])(
        train=None, 
        test=val,
        root=config['ROOT'],
        split=config['SPLIT']['TEST'],
        base_size=513,
        crop_size=config['IMAGE']['SIZE']['TEST'],
        mean=(config['IMAGE']['MEAN']['B'], config['IMAGE']['MEAN']['G'], config['IMAGE']['MEAN']['R']),
        warp=config['WARP_IMAGE'],
        scale=None,
        flip=False
    )

te = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=1,
        num_workers=config['NUM_WORKERS'],
        shuffle=False,
    )

class_emb = get_embedding(config)
class_emb_vis = class_emb[visible_classes]
class_emb_vis_ = torch.zeros((config['ignore_index'] + 1 - class_emb_vis.shape[0], class_emb_vis.shape[1]), dtype = torch.float32)
class_emb_vis_aug = torch.cat((class_emb_vis, class_emb_vis_), dim=0).cuda()
class_emb_all = class_emb[visible_classes_test].cuda()
print (class_emb_all.size())

Em = nn.Embedding.from_pretrained(class_emb_all)
Em = Em.cuda()
Em.weight.requires_grad = False

label_count = config['dis']['out_dim_cls'] - 1 # 20 for voc12
print(label_count)

net = Net(label_count, True)
net.cuda()
net = torch.nn.DataParallel(net)

torch.backends.cudnn.enabled = True

block_size = 5
roll_times = 2
display_iter = 10

# Only test
if args.val == True:
    assert(args.load_model != 'None')
    assert(os.path.exists(args.load_model))
    state_dict = torch.load(args.load_model)['state_dict']
    net.load_state_dict(state_dict, strict=True)
    print('\n*** Testing model {0} ... ***'.format(args.load_model))
    logger.write('\n*** Testing model {0} ... ***'.format(args.load_model))

    # compute error on test set
    err_te = []
    net.train(False)

    with torch.no_grad():
        for _, gt, _ in te:
            # do gt mapping for test, since no visibility_mask for test
            gt = torch.from_numpy(cls_map_test[gt]).long()

            H, W = gt.shape[1], gt.shape[2]
            assert (H >= block_size and W >= block_size)

            hh = H // block_size
            ww = W // block_size
            num_blocks_all = hh * ww

            gt0 = gt[:, :hh * block_size, :ww * block_size].cuda()
            gt1 = gt0.view(gt.shape[0], hh, block_size, ww, block_size)
            gt2 = gt1.permute(0,1,3,2,4).contiguous()
            gt3 = gt2.view(gt.shape[0]*num_blocks_all, block_size, block_size)
            gt4 = gt3.view(-1, block_size**2)
            gt5 = torch.where(gt4==config['ignore_index'], torch.zeros(gt4.shape).long().cuda(), torch.ones(gt4.shape).long().cuda()).long()
            gt6 = torch.sum(gt5, dim=1).long()
            indices = torch.where(gt6!=0)[0].long()
            if indices.shape == torch.Size([0]):
                continue
            gt_sample = gt3[indices,:,:]

            target = gt_sample.clone().cuda().contiguous()

            label = gt_sample
            ### label[label==config['ignore_index']]=0 !!! wrong
            # using np.roll to fill in holes
            label_numpy = label.cpu().numpy().astype(np.int64)
            for rt in range(roll_times):
                data_, mask_ = fill_in_hole(
                    a=label_numpy,
                    fill_value=config['ignore_index'], 
                    sh=(-1,1), 
                    ax=(1,2)  # roll on H(dim=1) and W(dim=2) of label_numpy
                )
                label_numpy = data_
            assert(not(mask_.any()==True))

            label = torch.from_numpy(label_numpy).long().cuda()
            assert(label.max()<label_count)

            input = Em(label).permute(0,3,1,2).contiguous()
            h = Em(label).float().contiguous()#.cuda()

            pred = net(input,h)
            loss = F.cross_entropy(pred, target, ignore_index=config['ignore_index'])
            err_te.append(loss.item())

    # print test results
    print ('nll_te={:.7f}\n'.format(np.mean(err_te)))
    logger.write('nll_te={:.7f}\n'.format(np.mean(err_te)))

    sys.exit(0)


# Train/val
if args.load_model != 'None':
    assert(os.path.exists(args.load_model))
    state_dict = torch.load(args.load_model)['state_dict']
    net.load_state_dict(state_dict, strict=True)
    print("Successfully loading preTrain model {0}".format(args.load_model))

optimizer = optim.Adam(net.parameters(),lr=0.000001)

for epoch in range(args.resume_from, 10):

    # train
    err_tr = []
    cuda.synchronize()
    time_tr = time.time()
    net.train(True)

    iteration = 0
    
    for _, gt in tr:
        H, W = gt.shape[1], gt.shape[2]
        assert (H >= block_size and W >= block_size)

        hh = H // block_size
        ww = W // block_size
        num_blocks_all = hh * ww

        gt0 = gt[:, :hh * block_size, :ww * block_size].cuda()
        gt1 = gt0.view(gt.shape[0], hh, block_size, ww, block_size)
        gt2 = gt1.permute(0,1,3,2,4).contiguous()
        gt3 = gt2.view(gt.shape[0]*num_blocks_all, block_size, block_size)
        gt4 = gt3.view(-1, block_size**2)
        gt5 = torch.where(gt4==config['ignore_index'], torch.zeros(gt4.shape).long().cuda(), torch.ones(gt4.shape).long().cuda()).long()
        gt6 = torch.sum(gt5, dim=1).long()
        indices = torch.where(gt6!=0)[0].long()
        if indices.shape == torch.Size([0]):
            continue
        gt_sample = gt3[indices,:,:]

        target = gt_sample.clone().cuda().contiguous()

        label = gt_sample
        ### label[label==config['ignore_index']]=0 !!! wrong
        # using np.roll to fill in holes
        label_numpy = label.cpu().numpy().astype(np.int64)
        for rt in range(roll_times):
            data_, mask_ = fill_in_hole(
                a=label_numpy,
                fill_value=config['ignore_index'], 
                sh=(-1,1), 
                ax=(1,2)  # roll on H(dim=1) and W(dim=2) of label_numpy
            )
            label_numpy = data_
        assert(not(mask_.any()==True))

        label = torch.from_numpy(label_numpy).long().cuda()
        assert(label.max()<label_count)

        input = Em(label).permute(0,3,1,2).contiguous()
        h = Em(label).float().contiguous()#.cuda()
            
        pred = net(input,h)
        #print(target)
        #print(pred)
        loss = F.cross_entropy(pred, target, ignore_index=config['ignore_index'])
        err_tr.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % display_iter == 0:
            print('epoch-[{0:0>3d}] iter-[{1:0>6d}] err_tr-[{2:.7f}]'.format(epoch+1, iteration, np.mean(err_tr)))
            logger.write('epoch-[{0:0>3d}] iter-[{1:0>6d}] err_tr-[{2:.7f}]'.format(epoch+1, iteration, np.mean(err_tr)))

            model_name = os.path.join('blocks_new_5', 'example.pth')
            torch.save(
                    {
                        'epoch': 0,
                        'state_dict': net.state_dict()
                    },
                    model_name
                )
            print('ok')
            sys.exit(0)
    
    cuda.synchronize()
    time_tr = time.time() - time_tr

    # save models
    model_name = os.path.join('blocks_new_5', '{0}_{1:0>3d}.pth'.format(config['dataset'], epoch + 1))
    torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': net.state_dict()
            },
            model_name
        )

    # print train results
    print ('epoch={}; nll_tr={:.7f}; time_tr={:.1f}s'.format(
        epoch+1, np.mean(err_tr), time_tr))
    logger.write('epoch={}; nll_tr={:.7f}; time_tr={:.1f}s'.format(
        epoch+1, np.mean(err_tr), time_tr))


    # compute error on test set
    err_te = []
    time_te = time.time()
    net.train(False)

    with torch.no_grad():
        for _, gt, _ in te:
            # do gt mapping for test, since no visibility_mask for test
            gt = torch.from_numpy(cls_map_test[gt]).long()

            H, W = gt.shape[1], gt.shape[2]
            assert (H >= block_size and W >= block_size)

            hh = H // block_size
            ww = W // block_size
            num_blocks_all = hh * ww

            gt0 = gt[:, :hh * block_size, :ww * block_size].cuda()
            gt1 = gt0.view(gt.shape[0], hh, block_size, ww, block_size)
            gt2 = gt1.permute(0,1,3,2,4).contiguous()
            gt3 = gt2.view(gt.shape[0]*num_blocks_all, block_size, block_size)
            gt4 = gt3.view(-1, block_size**2)
            gt5 = torch.where(gt4==config['ignore_index'], torch.zeros(gt4.shape).long().cuda(), torch.ones(gt4.shape).long().cuda()).long()
            gt6 = torch.sum(gt5, dim=1).long()
            indices = torch.where(gt6!=0)[0].long()
            if indices.shape == torch.Size([0]):
                continue
            gt_sample = gt3[indices,:,:]

            target = gt_sample.clone().cuda().contiguous()

            label = gt_sample
            ### label[label==config['ignore_index']]=0 !!! wrong
            # using np.roll to fill in holes
            label_numpy = label.cpu().numpy().astype(np.int64)
            for rt in range(roll_times):
                data_, mask_ = fill_in_hole(
                    a=label_numpy,
                    fill_value=config['ignore_index'], 
                    sh=(-1,1), 
                    ax=(1,2)  # roll on H(dim=1) and W(dim=2) of label_numpy
                )
                label_numpy = data_
            assert(not(mask_.any()==True))

            label = torch.from_numpy(label_numpy).long().cuda()
            assert(label.max()<label_count)

            input = Em(label).permute(0,3,1,2).contiguous()
            h = Em(label).float().contiguous()#.cuda()

            pred = net(input,h)
            loss = F.cross_entropy(pred, target, ignore_index=config['ignore_index'])
            err_te.append(loss.item())

    cuda.synchronize()
    time_te = time.time() - time_te

    # print test results
    print ('epoch={}; nll_te={:.7f}; time_te={:.1f}s'.format(
        epoch+1, np.mean(err_te), time_te))
    logger.write('epoch={}; nll_te={:.7f}; time_te={:.1f}s'.format(
        epoch+1, np.mean(err_te), time_te))

