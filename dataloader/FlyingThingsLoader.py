from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import re
import random


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) 
    
def load_disparity(file_path):
    return readPFM(file_path)

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def preprocess_data(image, augment=False):

    if augment:
        print('TODO Augmentation')

    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    return data_transforms(image)

class FlyingThingsDataloader(Dataset):

    def __init__(self, left_images, right_images, left_disparities, train):
        self.left_images = left_images
        self.right_images = right_images
        self.left_disparities = left_disparities
        self.train = train

    def __getitem__(self, index):

        left_image_path = self.left_images[index]
        right_image_path = self.right_images[index]
        disparity_path = self.left_disparities[index]

        left_img = load_image(left_image_path)#/256
        right_img = load_image(right_image_path)#/256
        data, _ = readPFM(disparity_path)
        data = np.ascontiguousarray(data, dtype=np.float32)#/256

        if self.train:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            data = data[y1:y1 + th, x1:x1 + tw]

            left_img = preprocess_data(left_img)
            right_img = preprocess_data(right_img)

            return left_img, right_img, data
        else:

            w, h = left_img.size
            left_img = left_img.crop((w - 960, h - 544, w, h))
            right_img = right_img.crop((w - 960, h - 544, w, h))

            left_img = preprocess_data(left_img)
            right_img = preprocess_data(right_img)

            return left_img, right_img, data

    def __len__(self):
        return len(self.left_images)

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
'''
if __name__ == '__main__':

    im = load_image('sample_dataset/RGB_cleanpass/left/0006.png')
    print(im.size)
    data, scale = load_disparity("sample_dataset/disparity/0006.pfm")
    data = np.asarray(data)
    print(data.shape)'''

