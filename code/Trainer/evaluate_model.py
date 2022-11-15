import os
import tensorflow as tf
import itertools
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import BinaryAccuracy
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class Evaluate:

    def __init__(self, Dataset, class_num, class_label):
        self.path = os.path.join(os.getcwd(), "saved_model")
        self.dataset = Dataset
        self.class_num = class_num
        self.class_label = class_label

    def evaluate_model(self):

        model = load_model(os.path.join(self.path, 'model.h5'))
        loss, acc = model.evaluate(self.dataset["test_X"], self.dataset["test_Y"])
        print("loss : ", loss, " / acc : ", acc)

        pred = model.predict(self.dataset["test_X"])
        self.plot_confusion_matrix(label=self.dataset["test_Y"], predit=pred)
        self.get_scores(target=self.dataset["test_Y"], predictions=pred)


    def get_scores(self, target, predictions):

        target_label = np.argmax(target, axis=1)
        predit_label = np.argmax(predictions, axis=1)

        file = open(os.path.join(self.path, "score.txt"), 'w')
        file.write(classification_report(target_label, predit_label))
        file.close()

        return


    def plot_confusion_matrix(self, label, predit, target_names=None, cmap=None, normalize=True,
                              title='Confusion matrix'):

        labels = self.class_label
        target_label = np.argmax(label, axis=1)
        predit_label = np.argmax(predit, axis=1)
        cm = confusion_matrix(target_label, predit_label)

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
        if cmap is None:
            cmap = plt.get_cmap('Blues')
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        c = plt.colorbar()
        plt.clim(0, 1)

        plt.title(title)
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names)
            plt.yticks(tick_marks, target_names)
        if labels:
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if normalize:
                    plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")
                else:
                    plt.text(j, i, "{:,}".format(cm[i, j]),
                             horizontalalignment="center",
                             color="white" if cm[i, j] > thresh else "black")

        plt.xticks(np.arange(8), labels)
        plt.yticks(np.arange(8), labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig(os.path.join(self.path, "cm.png"))
