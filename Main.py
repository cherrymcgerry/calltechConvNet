from Data.Setup_Database import setup_database
from Model.Model import Model
import os
import traceback
import pickle



def main():
    data_loader = setup_database(True, True, 20)

    with open(os.path.join('../101_ObjectCategories', 'label_dictionary.data'), 'rb') as f:
        labels = pickle.load(f)

    for label in labels:
        print(F'{labels.index(label)} , {label["label"]}')

    model = Model(data_loader)

    model.train()


if __name__ == '__main__':
    print("python code started")
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
