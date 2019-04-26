import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms, datasets

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', default='./images', help='the path of input images')
parser.add_argument('--output_path', default='./output', help='the path of output images')
parser.add_argument('--label_file', default='./labels.csv', help='the file of image labels')
parser.add_argument('--iterative', default=False, action='store_true', help='implement iterative FGSM')

def read_file(file_name):
    df = pd.read_csv(file_name)
    label = df.values[:,3].reshape(-1, 1).astype(np.int)

    return label

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.file_list = []
        file_list = os.listdir(self.path)
        for file in file_list:
            file_ext = os.path.splitext(file)[-1]
            if file_ext == '.png':
                self.file_list.append(file)
        self.file_list.sort()
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, index):
        img = default_loader(os.path.join(self.path, self.file_list[index]))
        if self.transform is not None:
            img = self.transform(img)
        return img

def main(input_path, output_path, label_file, iterative):
    model = models.resnet50(pretrained=True).cuda()
    model.eval()

    labels = read_file(label_file)
    labels = torch.tensor(labels).cuda()

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    inverse = transforms.Compose([
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    ])

    img_dataset = ImageDataset(input_path, transform=transform)
    img_loader = DataLoader(img_dataset)

    adv_imgs = []

    criterion = nn.CrossEntropyLoss()

    if iterative:
        max_try = 25
    else:
        max_try = 1



    tolerence = 0.0225
    epsilon = 0.03 / 20

    for index, batch in enumerate(img_loader):
        img = batch
        min_val = torch.min(img) - tolerence
        max_val = torch.max(img) + tolerence
        
        img.requires_grad = True

        label = labels[index]

        for i in range(max_try):
            model.zero_grad()

            output = model(img.cuda())
            loss = criterion(output, label)
            loss.backward()

            grad = torch.sign(img.grad.data)
            img = img.data + epsilon * grad
            img = torch.clamp(img, min_val, max_val)
            img = torch.tensor(img.data.cpu().numpy(), requires_grad=True)

            pred_label = torch.argmax(model(img.cuda()))

            if pred_label != label and i%5 == 0:
                #print('{} , {}'.format(pred_label, label))
                print('Successfully Attacked. {}'.format(index))
                break

        adv_imgs.append(img)

    ToPILImage = transforms.ToPILImage()
    for index, img in enumerate(adv_imgs):
        img = img.squeeze()
        img = inverse(img)        
        img = ToPILImage(img)
        file_name = '{:0>3d}.png'.format(index)
        img.save(os.path.join(output_path, file_name))

if __name__ == '__main__':
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    label_file = args.label_file
    iterative = args.iterative
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    main(input_path, output_path, label_file, iterative)