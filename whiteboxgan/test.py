from tensorboardX import SummaryWriter
import time
from torchvision import transforms
import torchvision.utils as vutils
import torch
import numpy as np
from PIL import Image
def process(img):
    img=torch.from_numpy(np.array(img).transpose((2,0,1))).unsqueeze(0)
    return img

writer = SummaryWriter('log')
img=Image.open('data/fake/2013-11-23 13_03_15.jpg')
img=process(img)

img2=Image.open('data/fake/2013-11-10 12_45_41.jpg')
img2=process(img2)
x = vutils.make_grid(torch.cat((img,img2),0), scale_each=True)
writer.add_image('img2',x, global_step=1)