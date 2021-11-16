from network.net_build import *
from data_vis import plot_img_and_mask

import logging
import os
import sys
from glob import glob
from guild_filter_code import guide_filter
import torch.nn as nn
from torch import optim
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

def resize(img):
    a = img
    if len(a.shape) == 2:
        a = np.expand_dims(a, axis=2)
    a = a.transpose((2, 0, 1))
    if a.max() > 1:
        a = a/255
    return a

net_path='./checkpoints/checkpointD_c2_epoch27.pth'
predict_pic_path='./test_images/1s.jpg'

def pre_dict(net,
             full_img,
             device):
    net.eval()
    
    img=torch.from_numpy(resize(full_img)).type(torch.FloatTensor)
    
    img=img.unsqueeze(0)
    img=img.to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output=net(img)
        output=output.squeeze(0)
    return output.cpu()

if __name__=="__main__":
    net = generator(n_channels=3, n_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    for i in range(1,2): 
        net.load_state_dict(torch.load(net_path.format(i), map_location=device))
        img = Image.open(predict_pic_path)
        img=np.array(img)
        h, w ,c= np.shape(img)
        h, w = (h//8)*8, (w//8)*8
        img = img[:h, :w ,:]
        out=pre_dict(net,img,device).squeeze(0)
        out=(out*255)
        out=np.clip(np.asarray(out), 0, 255)
        out=guide_filter(torch.from_numpy(img.transpose((2, 0, 1))).type(torch.FloatTensor).unsqueeze(0),torch.from_numpy(out).unsqueeze(0), r=1, eps=5e-3).squeeze(0)
        out=out.permute((1, 2, 0))
        out=np.asarray(out).astype(np.uint8)
        out_image=Image.fromarray(out)
        out_image.save('./res/result{}.jpg'.format(i))

