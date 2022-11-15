import random

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

class Datamodule:

    def __init__(self, class_list, train_num:int=120):
        self.path = os.path.join(os.getcwd(), "raw_data")
        self.train_num = train_num
        self.class_list = class_list


    def random_idx(self, dataset, labels):
        idx = list(range(len(labels)))
        random.shuffle(idx)
        return dataset[idx], labels[idx]


    def load_data(self):
        self.dataset = np.load(os.path.join(self.path, "dataset.npy"))
        self.label = np.load(os.path.join(self.path, "labels.npy"))
        self.data_shape = (1, self.dataset.shape[1], self.dataset.shape[2], self.dataset.shape[3])
        self.unique_class = np.unique(self.label)


    def split_train_test_set(self):

        if os.path.isfile(os.path.join(self.path, "train_dataset.npy")):
            train_dataset = np.load(os.path.join(self.path, "train_dataset.npy"))
            train_labels_onehot = np.load(os.path.join(self.path, "train_labels.npy"))
            test_dataset = np.load(os.path.join(self.path, "test_dataset.npy"))
            test_labels_onehot = np.load(os.path.join(self.path, "test_labels.npy"))

        else:
            self.load_data()

            train_dataset = np.zeros(self.data_shape)
            test_dataset = np.zeros(self.data_shape)
            train_labels, test_labels = np.zeros(0, dtype='<U26'), np.zeros(0, dtype='<U26')
            idx_list = []
            for i, cn in enumerate(self.class_list):
                idx_list.append(np.where(self.label == cn)[0])
                idx_train = self.train_num

                class_dataset = self.dataset[idx_list[i][0]:idx_list[i][-1] + 1]
                class_labels  = self.label[idx_list[i][0]:idx_list[i][-1] + 1]
                class_dataset, class_labels = self.random_idx(class_dataset, class_labels)
                train_dataset = np.concatenate((train_dataset, class_dataset[:idx_train]), axis=0)
                train_labels = np.concatenate((train_labels, class_labels[:idx_train]))
                test_dataset = np.concatenate((test_dataset, class_dataset[idx_train:]), axis=0)
                test_labels = np.concatenate((test_labels, class_labels[idx_train:]))

            train_dataset = train_dataset[1:] / 255.0
            test_dataset = test_dataset[1:] / 255.0

            enc = OneHotEncoder()
            train_labels_new = train_labels.reshape(-1, 1)
            test_labels_new = test_labels.reshape(-1, 1)

            enc.fit(train_labels_new)
            train_labels_onehot = np.array(enc.transform(train_labels_new).toarray())
            test_labels_onehot = np.array(enc.transform(test_labels_new).toarray())

            np.save(os.path.join(self.path, "train_dataset.npy"), train_dataset)
            np.save(os.path.join(self.path, "train_labels.npy"), train_labels_onehot)
            np.save(os.path.join(self.path, "test_dataset.npy"), test_dataset)
            np.save(os.path.join(self.path, "test_labels.npy"), test_labels_onehot)

        print("train data shape = ", train_dataset.shape)
        print("train label shape = ", train_labels_onehot.shape)
        print("test data shape = ", test_dataset.shape)
        print("test label shape = ", test_labels_onehot.shape)

        return {"train_X": train_dataset, "train_Y": train_labels_onehot, "test_X": test_dataset, "test_Y": test_labels_onehot}