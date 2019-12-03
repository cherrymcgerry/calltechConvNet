import os

from torch.utils.data import DataLoader
from dataset.dataset import data
from dataset.preprocess import preprocess

#Sets up a data loader for the requested split
def setup_database(args, split="train"):
    #Choose correct file depending on desired split
    file = ('data.data')
    path = os.path.join(args.dataroot, file)
    #If file not present it may still need to be preprocessed
    if not os.path.isfile(path):
        print("Pre-processing dataset")
        preprocess(args.dataroot)

    #Initialise dataset and setup a data loader
    dataset = data(path, split)
    dataLoader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, num_workers=2)

    return dataLoader
