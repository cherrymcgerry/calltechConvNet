import os
import pickle

def preprocess(dataroot):
    directories = os.listdir(dataroot)
    directories.sort()
    train = []; val = []; test = []; data =[]
    for i in range(0,len(directories)):
        dir = os.path.join(dataroot,directories[i])
        files = os.listdir(dir)
        for file in files:
            data.append([os.path.join(dir,file), i])
    train.append(data[:int(0.64*len(data))])
    val.append(data[int(0.64*len(data)):int(0.8*len(data))])
    test.append(data[int(0.8*len(data)):])


    dict = {'train': train,
            'val':   val,
            'test':  test}

    #Writing db to file
    with open(os.path.join(dataroot,'data.data' ), 'wb') as f:
        pickle.dump(dict, f)
