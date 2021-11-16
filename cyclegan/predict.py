from model.net_build import *
import torch
import numpy as np
from utils import *
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

net_path='./checkpoints/checkpointGab_1_epoch{}.pth'
predict_pic_path='./test_images/2014-08-03 09:47:19.jpg'

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([    transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])
    Gab=generator(n_channels=3, n_classes=3).to(device=device)
    for i in range(1,11):
        Gab.load_state_dict(torch.load(net_path.format(i), map_location=device))
        img = Image.open(predict_pic_path)
        img = np.array(img)
        h, w ,c= np.shape(img)
        h, w = (h//8)*8, (w//8)*8
        img = img[:h, :w ,:]
        Gab.eval()
        img=transform(img).type(torch.FloatTensor)
        img=img.unsqueeze(0)
        img=img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output=Gab(img)
            output=output.squeeze(0)
            save_image(output,'./res/result{}.jpg'.format(i))