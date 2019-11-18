import numpy as np
import random
import os
import pickle
import cv2 as cv
import torch
import pickle
from skimage import io, transform

#OUTPUT_SIZE = (100, 100)



def preprocess(dataRoot, normalise=False):
    # get images and corresponding labels in a dictionary
    dataSamples = []
    dataTrainSamples = []
    dataTestSamples = []
    labels = []
    print('iterating dataRoot')
    for root, dirs, files in os.walk(dataRoot, topdown=True):
        for name in dirs:
            for rootf, dirf, filef in os.walk(os.path.join(dataRoot, name), topdown=True):
                for filen in filef:
                    image = io.imread(os.path.join(dataRoot, name, filen), as_gray=True)
                    label = name
                    dataSamples.append({'image': np.array(image), 'label': label})

                # to later convert labels to one hot vectors
                if not labels.__contains__(name):
                    labels.append(name)
        for sample in dataSamples[:int(0.8*len(dataSamples))]:
            dataTrainSamples.append(sample)
        for sample in dataSamples[int(0.8*len(dataSamples)):]:
            dataTestSamples.append(sample)
        dataSamples = []


    print('converting labels to onehotvectors')
    # converts label names to one hot vectors and saves dictionary
    labels = to_one_hot_vector(labels)
    with open(os.path.join(dataRoot, 'label_dictionary.data'), 'wb') as f:
        pickle.dump(labels, f)

    print('converting sample labels to one hot vectors')
    # converts dataSampleLabels to one hot vectors according to dictionary
    dataTrainSamples = sample_labels_to_one_hot_vector(dataTrainSamples, labels)
    dataTestSamples = sample_labels_to_one_hot_vector(dataTestSamples, labels)


    print('applying transforms')
# Transforms
    # rescale
    # randomCrop
    # toTensor
    #for sample in dataTrainSamples:
        #rescale(sample['image'], OUTPUT_SIZE)
        #randomCrop(image, CROPOUTPUTSIZE)
        #ToTensor(sample['image'])


    #for sample in dataTestSamples:
        #rescale(sample['image'], OUTPUT_SIZE)
        # randomCrop(image, CROPOUTPUTSIZE)
        #ToTensor(sample['image'])


    print('shuffling data')
#shuffle data
    random.shuffle(dataTrainSamples)
    random.shuffle(dataTestSamples)

    print('saving train and test data')
#writing database to file
    with open(os.path.join(dataRoot, 'train_data.data'), 'wb') as f:
        pickle.dump(dataTrainSamples, f)

    with open(os.path.join(dataRoot, 'test_data.data'), 'wb') as f:
        pickle.dump(dataTestSamples, f)








def ToTensor(image):
    image = torch.from_numpy(image)
    return image



#def randomCrop(image, cropOutputSize):



# output_size (tuple or int): Desired output size. If tuple, output is
# matched to output_size. If int, smaller of image edges is matched
# to output_size keeping aspect ratio the same.



def to_one_hot_vector(labels):
    vectors = np.eye(labels.__len__())
    for i in range(labels.__len__()):
        labels[i] = {"label": labels[i], "vector": vectors[i]}
    return labels


def sample_labels_to_one_hot_vector(dataSamples, labels):
    for sample in dataSamples:

        for label in labels:
            if label['label'] == sample['label']:
                sample['label'] = label['vector']
    return dataSamples
