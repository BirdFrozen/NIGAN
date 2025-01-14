import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil
from comMIOU import computeall

from time import time
from pathlib import Path

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool,DinkNet34_lzy,DinkNet101_lzy

from tqdm import tqdm

BATCHSIZE_PER_CARD = 4
torch.cuda.set_device(1)


class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        # self.net = torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))
        
    def test_one_img_from_path(self, path, evalmode = True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(path)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(path)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(path)

    def test_one_img_from_path_8(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2

    def test_one_img_from_path_4(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2
    
    def test_one_img_from_path_2(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, path):
        img = cv2.imread(path)#.transpose(2,0,1)[None]
        # print(path)
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        
#source = 'dataset/test/'
def test_all(model_dir):
    # _base_dir = '/home/mj/data/work_road/data/SpaceNet'
    # _image_dir = os.path.join(_base_dir, 'RGB-PanSharpen_8bit')
    # _cat_dir = os.path.join(_base_dir, 'masks_3m')

    # split = ['test']
    # _splits_dir = os.path.join(_base_dir)

    # im_ids = []
    # images = []
    # categories = []

    # for splt in split:
    #     with open(os.path.join(os.path.join(_splits_dir, f'{splt}' + '.txt')), "r") as f:
    #         lines = f.read().splitlines()

    #     for ii, line in enumerate(lines):
    #         _image = os.path.join(_image_dir, line)
    #         _cat = os.path.join(_cat_dir, line.split('n_')[1])
    #         assert os.path.isfile(_image)
    #         assert os.path.isfile(_cat)
    #         im_ids.append(line)
    #         images.append(_image)
    #         categories.append(_cat)

    source = '/home/mj/data/work_syj/code/GeoSeg/data/vaihingen/test/images/'
    val = os.listdir(source)
    model_name = str(model_dir)
    solver = TTAFrame(DinkNet34_lzy)
    solver.load('/home/mj/data/work_syj/code/NIGAN/weights/' + model_name + '.th')
    tic = time()
    target = '/home/mj/data/work_syj/code/NIGAN/submits/' + model_name + '/'
    # os.mkdir(target)
    filepath = Path(target)
    if filepath.exists():
        shutil.rmtree(target, True)
        print('删除文件夹成功:{}'.format(target))
        os.mkdir(target)
    else:
        os.mkdir(target)

    print('test')
    for i, name in tqdm(enumerate(val),total = len(val)):
        # print('name:',name)
        # if i % 10 == 0:
        #     print(i / 10, '    ', '%.2f' % (time() - tic))
        mask = solver.test_one_img_from_path(source + name)
        mask[mask > 4.0] = 255
        mask[mask <= 4.0] = 0
        # mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
        cv2.imwrite(target + name[:-4] + '.png', mask.astype(np.uint8))
    labledir = '/home/mj/data/work_syj/code/GeoSeg/data/vaihingen/test/masks/'
    predir = '/home/mj/data/work_syj/code/NIGAN/submits/' + model_name + '/'
    MIOU,IOU,F1 = computeall(labledir, predir)
    return MIOU,IOU,F1


if __name__ == "__main__":
    source = '/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/test/'
    val = os.listdir(source)
    model_name = 'DinkNet34_lzy_01_08_08_15_48_24_zero_2000_00002_out_3d_bestmiou'
    solver = TTAFrame(DinkNet34_lzy)
    solver.load('/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/weights/'+model_name+'.th')
    tic = time()
    target = '/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/submits/'+model_name+'/'
    # os.mkdir(target)
    filepath = Path(target)
    if filepath.exists():
        shutil.rmtree(target, True)
        print('删除文件夹成功:{}'.format(target))
        os.mkdir(target)
    else:
        os.mkdir(target)

    for i, name in enumerate(val):
        # print('name:',name)
        if i % 10 == 0:
            print(i/10, '    ', '%.2f' % (time()-tic))
        mask = solver.test_one_img_from_path(source+name)
        mask[mask > 4.0] = 255
        mask[mask <= 4.0] = 0
        mask = np.concatenate([mask[:,:,None],mask[:,:,None], mask[:, :, None]], axis=2)
        cv2.imwrite(target+name[:-7]+'mask.png',mask.astype(np.uint8))
    labledir = '/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/test_out/'
    predir = '/run/media/cug/00077CE80009E4AD/Liuzhuoyue/data/road_seg/submits/'+model_name+'/'
    computeall(labledir, predir)
