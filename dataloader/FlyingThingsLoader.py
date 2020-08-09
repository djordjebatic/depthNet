import torch
import numpy as np
from utils.python_pfm import readPFM
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

def load_disparity(file_path):
    pfm, _ = readPFM(file_path)
    data = np.ascontiguousarray(pfm, dtype=np.float32)
    return data

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

        left_img = load_image(left_image_path)
        right_img = load_image(right_image_path)

        w, h = left_img.size

        data = load_disparity(disparity_path)

        if self.train:
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            #data = np.negative(data)
            data = data[y1:y1 + th, x1:x1 + tw]

        else:
            th, tw = 512, 960

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            data = data[y1:y1 + th, x1:x1 + tw]

        left_img = preprocess_data(left_img)
        right_img = preprocess_data(right_img)

        return left_img, right_img, data

    def __len__(self):
        return len(self.left_images)