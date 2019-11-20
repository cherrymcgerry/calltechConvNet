import pickle
import os
import torch
from skimage import io, transform
from torchvision.transforms import transforms

from torch.utils.data import Dataset

OUTPUT_SIZE = (50, 50)

class data(Dataset):

    def __init__(self, dataroot, is_train, device='cpu', normalize=True):
        path = os.path.join(dataroot, 'train_data.data' if is_train else 'test_data.data')
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

        self.normalize = normalize
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputI = self.data[idx]['image']
        output = self.data[idx]['label']

        inputI = rescale(inputI, OUTPUT_SIZE)
        inputI = torch.from_numpy(inputI)
        inputI = inputI.view(1, 50, 50)

        if self.normalize:
            composed = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
            composed(inputI)

        return inputI, output


def rescale(image, output_size):
    assert isinstance(output_size, (int, tuple))

    h, w = image.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    image = transform.resize(image, (new_h, new_w))
    return image