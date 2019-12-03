import torch
import torch.nn as nn
import torch.optim as optim
from Model.InitModel import initModel
from Model.ConvNet1 import ConvNet
from Model.InitModel import saveCheckpoint
from Data.Setup_Database import setup_database
import xlsxwriter
import os
import pickle

EPOCHS = 100
CHECKPOINT_FREQ = 10


class Model(object):
    def __init__(self, data_loader):
        print("setting up model")

        # setup test data
        self.testData_loader = setup_database(False, True, 5)

        # Setup device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.dict = {}
        self.epoch = 0
        self.excelInitialized = False
        self.data_loader = data_loader
        self.loss = []
        self.result_root = './result'

        # inputI, output = next(iter(data_loader))
        # size = [inputI.size(), output.size()]
        self.model = ConvNet()

        self.optim = optim.Adam(self.model.parameters(), lr=0.0001)

        initModel(self, self.device)

        # TODO EVAL

    def test(self):
        correct = 0
        total = 0
        self.model.eval()
        for i, data in enumerate(self.testData_loader):
            with torch.no_grad():

                inputI = data[0].view(-1, 1, 50, 50).to(device=self.device, dtype=torch.float)  # TODO add .view
                output = data[1].to(device=self.device, dtype=torch.float)

                prediction = self.model(inputI)

            # eval
            eval = []
            predicted_class = []
            real_class = []
            for sample in prediction:
                predicted_class.append(torch.argmax(sample))
            for sample in output:
                real_class.append(torch.argmax(sample))

            for i in range(len(predicted_class) - 1):
                eval.append({'pred': predicted_class[i], 'real': real_class[i]})

            for sample in eval:
                dictKeys = self.dict.keys()
                if sample['pred'] == sample['real']:
                    correct += 1
                    for key in dictKeys:
                        if sample['real'] == key:
                            self.dict[key] = {'correct': (self.dict.get(key)).get('correct'),
                                              'total': self.dict.get(key).get('total'),
                                              'testCorrect': self.dict.get(key).get('testCorrect') + 1,
                                              'testTotal': self.dict.get(key).get('testTotal') + 1}
                            break
                else:
                    for key in dictKeys:
                        if sample['real'] == key:
                            self.dict[key] = {'correct': (self.dict.get(key)).get('correct'),
                                              'total': self.dict.get(key).get('total'),
                                              'testCorrect': self.dict.get(key).get('testCorrect'),
                                              'testTotal': self.dict.get(key).get('testTotal') + 1}
                            break
                total += 1

        # self.epoch += 1
        print(F'Test Accuracy: {round(correct / total, 3)}')
        self.model.train()

    def train(self):
        lossF = nn.BCELoss()
        # lossF = nn.MSELoss()
        evalArr = []

        print("starting training loop")
        while self.epoch < EPOCHS:
            print(F'epochs: {self.epoch}/{EPOCHS}')
            correct = 0
            total = 0
            for i, data in enumerate(self.data_loader):
                self.model.zero_grad()

                inputI = data[0].view(-1, 1, 50, 50).to(device=self.device, dtype=torch.float)  # TODO add .view
                output = data[1].to(device=self.device, dtype=torch.float)

                prediction = self.model(inputI)
                loss = lossF(prediction, output)

                loss.backward()
                self.optim.step()

                # eval
                eval = []
                predicted_class = []
                real_class = []
                for sample in prediction:
                    predicted_class.append(torch.argmax(sample))
                for sample in output:
                    real_class.append(torch.argmax(sample))

                for i in range(len(predicted_class) - 1):
                    eval.append({'pred': predicted_class[i], 'real': real_class[i]})

                for sample in eval:
                    update = {}

                    dictKeys = self.dict.keys()
                    if sample['pred'] == sample['real']:
                        correct += 1
                        indict = False
                        for key in dictKeys:
                            if sample['real'] == key:
                                self.dict[key] = {'correct': ((self.dict.get(key)).get('correct') + 1),
                                                  'total': ((self.dict.get(key).get('total')) + 1), 'testCorrect': 0,
                                                  'testTotal': 0}
                                indict = True
                                break
                        if not indict:
                            update = {sample['real']: {'correct': 1, 'total': 1, 'testCorrect': 0, 'testTotal': 0}}
                            self.dict.update(update)

                    else:
                        indict = False
                        for key in dictKeys:
                            if sample['real'] == key:
                                self.dict[key] = {'correct': (self.dict.get(key)).get('correct'),
                                                  'total': ((self.dict.get(key).get('total')) + 1), 'testCorrect': 0,
                                                  'testTotal': 0}
                                indict = True
                                break

                        if not indict:
                            update = {sample['real']: {'correct': 0, 'total': 1, 'testCorrect': 0, 'testTotal': 0}}
                            self.dict.update(update)

                    total += 1

            self.epoch += 1
            # print(torch.cuda.current_device())
            # print(torch.cuda.get_device_name(torch.cuda.current_device()))
            # print(torch.cuda.is_available())

            # accuracy + total, testaccuracy + total,
            print(F'Train Accuracy: {round(correct / total, 3)}')

            if self.epoch % CHECKPOINT_FREQ == 0:
                saveCheckpoint(self)
            self.test()
            for key, dict in sorted(self.dict.items()):
                print(
                    F'accuracy {key} : {round(dict["correct"] / dict["total"], 3)} total: {dict["total"]},  test: {round(dict["testCorrect"] / dict["testTotal"], 3)} testTotal: {dict["testTotal"]}')
            self.dataToExcel()
            self.dict = {}
        saveCheckpoint(self)
        print("Training finished")

    def getModel(self):
        return self.model

    def getEpoch(self):
        return self.epoch

    def dataToExcel(self):
        workbook = xlsxwriter.Workbook("results.xlsx")
        worksheet = workbook.add_worksheet()
        column = 1
        row = self.epoch + 2
        if self.excelInitialized:
            worksheet = self.initializeExcel(worksheet)

        # write train data

        worksheet.write(row, column, self.epoch)
        for key, dict in sorted(self.dict.items()):
            worksheet.write(row, column, round(dict["correct"] / dict["total"], 3))
            column += 1
        worksheet.write(row, column, dict["total"])
        column += 3
        for key, dict in sorted(self.dict.items()):
            worksheet.write(row, column, round(dict["testCorrect"] / dict["testTotal"], 3))
            column += 1
        workbook.close()


    def initializeExcel(self, worksheet):

        with open(os.path.join('../101_ObjectCategories', 'label_dictionary.data'), 'rb') as f:
            labels = pickle.load(f)
        row = 0
        column = 1

        for label in labels:
            worksheet.write(row, column, label['label'])
            worksheet.write(row + 1, column, labels.index(label))
            column += 1

        column = len(labels) + 3
        for label in labels:
            worksheet.write(row, column, label['label'])
            worksheet.write(row + 1, column, labels.index(label))
            column += 1

        self.excelInitialized = True
        return worksheet
