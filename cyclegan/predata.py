from os.path import splitext
from os import listdir

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
class Data_set(Dataset):
    def __init__(self, dir_fake, dir_real):
        self.dir_fake = dir_fake
        self.dir_real = dir_real
        self.name1 = [splitext(file)[0] for file in listdir(
            dir_fake) if not file.startswith('.')]
        self.name2 = [splitext(file)[0] for file in listdir(
            dir_real) if not file.startswith('.')]
        self.len1=len(self.name1)
        self.len2=len(self.name2)
        self.transform = transforms.Compose([   transforms.Resize(int(256*1.12), Image.BICUBIC), 
                                                transforms.RandomCrop(256), 
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])

    def __len__(self):
        return max(self.len1,self.len2)

    def __getitem__(self, i):
        file_fake = self.dir_fake + self.name1[i%self.len1] + '.jpg'
        file_real = self.dir_real + self.name2[i%self.len2] + '.jpg'
        try:
            fake = Image.open(file_fake)
            real = Image.open(file_real)
        except:
            print(file_fake)
            print(file_real)
        fake = self.transform(fake)
        real = self.transform(real)
        return {
            'fake': fake,
            'real': real,
        }  # return a dict
