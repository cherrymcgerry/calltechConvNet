import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

#Expansion of the Dataset class to fit our dataset
class data(Dataset):
    def __init__(self, path, split):

        self.transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
        transforms.RandomRotation(45),
        transforms.Resize((128,128)),
        transforms.RandomCrop((64,64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])
        #Open the file with data
        with open(path, 'rb') as f:
            self.data = pickle.load(f)[split][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #Load a specific data item
        input = Image.open(self.data[idx][0])
        #input = input.resize((64,64))
        #input = input.convert('L')
        input = self.transform(input)
        #plt.imshow(input)
        #plt.show()

        #Transform to torch tensor and to desired dimension and type
        #input = torch.from_numpy(input)
        #input = torch.unsqueeze(input,0).type(torch.FloatTensor)
        #output = torch.zeros(101)
        #output[self.data[idx][1]] = 1
        if True:
            output = torch.rand(101)/10
            output[self.data[idx][1]] = 1-torch.rand(1)/10

        return input, output
