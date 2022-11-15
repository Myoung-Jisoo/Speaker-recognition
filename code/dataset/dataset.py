import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import matplotlib.pyplot as plt

class Make_dataset:

    def __init__(self, class_list):
        self.path = os.path.join(os.getcwd(), "raw_data")
        self.sr = 16000
        self.frame_length = 0.025
        self.frame_stride = 0.010
        self.frame_size = 1
        self.input_nfft = int(round(self.sr * self.frame_length))
        self.input_stride = int(round(self.sr * self.frame_stride))

        self.classes = class_list


    def figure_to_array(self, fig):
        fig.canvas.draw()
        return np.array(fig.canvas.renderer._renderer)


    def make_mel(self, class_name, fn):
        # mel-spectrogram
        y, sr = librosa.load(os.path.join(self.path, class_name, fn), sr=self.sr)

        input_nfft = int(round(sr * self.frame_length))
        input_stride = int(round(sr * self.frame_stride))

        S = librosa.feature.melspectrogram(y=y, n_mels=40, n_fft=input_nfft, hop_length=input_stride)

        mel = librosa.power_to_db(S, ref=np.max)

        f = plt.figure(figsize=(2, 2))
        librosa.display.specshow(mel, y_axis=None, sr=sr, hop_length=input_stride, x_axis=None)
        plt.tight_layout()
        f_arr = self.figure_to_array(f)
        plt.close()

        return f_arr


    def make_dataset(self):
        data = []
        labels = []
        for i, class_name in enumerate(self.classes):
            print("class : ", class_name)
            cn_dir = os.path.join(self.path, class_name)
            for fn in tqdm(os.listdir(cn_dir)):
                mel = self.make_mel(class_name, fn)
                data.append(mel)
                labels.append(class_name)
        data = np.array(data)
        labels = np.array(labels)

        np.save(self.path + "/dataset.npy", data)
        np.save(self.path + "/labels.npy", labels)




