import librosa
import numpy as np
import pandas as pd


def feature_extract(files , label_num =None, label_len = None):

    if label_num is None and label_len is None :
        N = 0

        x_data = np.ndarray(shape=[0, 40], dtype=np.float32)

        y_data = np.ndarray(shape=[0, len(files)], dtype=np.float32)

        label_len = len(files)

        for file in files:

            X, sample_rate = librosa.load(file)

            mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)


            mfcc = np.transpose(mfcc)

            # label = np.full((mfcc.shape[0],1),N)
            # mfcc = np.append(mfcc, label, axis=1)


            label = np.zeros([1,label_len])
            label[0][N] = 1

            print('label', label,  label.shape)
            # print(x_data.shape)
            # print(mfcc.shape)
            x_data = np.append(x_data, mfcc, axis=0)

            print(y_data.shape)

            y_data = np.append(y_data, np.repeat(label, mfcc.shape[0], axis=0),axis=0)
            N = N + 1


            # print(x_data.shape)
            # print(y_data.shape)
            data = dict()
            data["X"] = x_data
            data["Y"] = y_data

        return data, x_data.shape[0]

    else :
        x_data = np.ndarray(shape=[0, 40], dtype=np.float32)

        y_data = np.ndarray(shape=[0, label_len], dtype=np.float32)

        for file in files:

            X, sample_rate = librosa.load(file)

            mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)

            mfcc = np.transpose(mfcc)

            # label = np.full((mfcc.shape[0],1),N)
            # mfcc = np.append(mfcc, label, axis=1)

            label = np.zeros([1, label_len])
            label[0][label_num] = 1

            print('label', label, label.shape)
            # print(x_data.shape)
            # print(mfcc.shape)
            x_data = np.append(x_data, mfcc, axis=0)

            print(y_data.shape)

            y_data = np.append(y_data, np.repeat(label, mfcc.shape[0], axis=0), axis=0)

            # print(x_data.shape)
            # print(y_data.shape)
            data = dict()
            data["X"] = x_data
            data["Y"] = y_data

        return data, x_data.shape[0]












