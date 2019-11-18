import torch
import torch.nn as nn
import torch.optim as optim
from Model.InitModel import initModel
from Model.ConvNet1 import  ConvNet
from Model.InitModel import saveCheckpoint

EPOCHS = 100
CHECKPOINT_FREQ = 10

class Model(object):
    def __init__(self, data_loader):
        print("setting up model")


        #Setup device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

        self.epoch = 0
        self.data_loader = data_loader
        self.loss = []
        self.result_root = './result'

        inputI, output = next(iter(data_loader))
        #size = [inputI.size(), output.size()]
        self.model = ConvNet()

        self.optim = optim.Adam(self.model.parameters(), lr=0.001)

        initModel(self, self.device)

        #TODO EVAL

    def train(self):
        lossF = nn.BCELoss()

        print("starting training loop")
        while self.epoch < EPOCHS:
            print(F'epochs: {self.epoch}/{EPOCHS}')
            correct = 0
            total = 0
            for i, data in enumerate(self.data_loader):
                self.model.zero_grad()

                inputI = data[0].view(-1,1,100,100).to(device=self.device, dtype=torch.float) #TODO add .view
                output = data[1].to(device = self.device, dtype=torch.float)

                prediction = self.model(inputI)
                loss = lossF(prediction, output)
                loss.backward()
                self.optim.step()


                #eval
                eval = []
                predicted_class = []
                real_class = []
                for sample in prediction:
                    predicted_class.append(torch.argmax(prediction))
                for sample in output:
                    real_class.append(torch.argmax(output))

                for i in range(len(predicted_class)):
                    eval[i] = {'pred': predicted_class[i], 'real': real_class[i]}

                for sample in eval:
                    if sample['pred'] == sample['real']:
                        correct += 1
                    total += 1

            self.epoch += 1
            print(F'Accuracy: {round(correct/total,3)}')
            if self.epoch % CHECKPOINT_FREQ == 0:
                saveCheckpoint(self)

        saveCheckpoint(self)
        print("Training finished")

    def getModel(self):
        return self.model

    def getEpoch(self):
        return self.epoch

