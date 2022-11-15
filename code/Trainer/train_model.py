import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt

from Trainer.cnn_model import cnn_model


class Trainer:

    def __init__(self, Dataset, class_num, input_shape, batch_size:int=30, epochs:int=100, valid_ratio:float=0.0):
        self.path = os.path.join(os.getcwd(), "saved_model")
        self.dataset = Dataset
        self.class_num = class_num
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.valid_ratio = valid_ratio


    def random_idx(self):
        idx = list(range(self.dataset["train_X"].shape[0]))
        random.shuffle(idx)
        self.dataset["train_X"] = self.dataset["train_X"][idx]
        self.dataset["train_Y"] = self.dataset["train_Y"][idx]


    def plot_acc_loss(self, hist):

        fig, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()

        loss_ax.plot(hist.history['loss'], 'y', label='train loss')
        if self.valid_ratio != 0:
            loss_ax.plot(hist.history['val_loss'], 'y', label='val loss')

        acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
        if self.valid_ratio != 0:
            acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')

        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')

        plt.savefig(os.path.join(self.path, "acc_loss.png"))
        plt.close()


    def train(self):

        self.random_idx()

        model = cnn_model(self.class_num, self.input_shape)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.005, patience=5, mode='min')
        model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'], )

        if self.valid_ratio != 0:
            history = model.fit(self.dataset["train_X"], self.dataset["train_Y"], validation_split=self.valid_ratio,
                                batch_size=self.batch_size, epochs=self.epochs, verbose=1, callbacks=[callback])
        else:
            history = model.fit(self.dataset["train_X"], self.dataset["train_Y"],
                                batch_size=self.batch_size, epochs=self.epochs, verbose=1, callbacks=[callback])

        model.save(os.path.join(self.path, "model.h5"))
        self.plot_acc_loss(history)

        loss, acc = model.evaluate(self.dataset["test_X"], self.dataset["test_Y"])
        print("loss : ", loss, " / acc : ", acc)
