import librosa
import os
import numpy as np
from scipy.io import wavfile
from librosa import stft, cqt




#    
# Using as the skeleton of the system
# This code was adapted from Github published on 19 Apr 2019
# accessed 15-7-2022
# https://github.com/kimmo1019/audioNet
# Code is modify for transforming to another spectrogram
# 
def preprocess(file_name, spec = 'mel', length=230000,): #transforming input data from waveform to spectrogram
    f, w_data = wavfile.read('./data/train/%s'%file_name)
    data = np.zeros(length,dtype='float32')
    if len(w_data)<=length:
        data[:len(w_data)] = w_data
    else:
        data = w_data[:length]
    data = data*1.0/np.max(data)  
    if spec == 'cqt':
        spectrogram = cqt(data, sr=44100, hop_length=128, fmin=80) #hop_lenth=128
    elif spec == 'stft':
        spectrogram = stft(data, n_fft=2048, hop_length=512)
    elif spec == 'mel':
        spectrogram = librosa.feature.melspectrogram(data, hop_length=256, n_fft=2048, n_mels=50) #n_fft=2048, hop_length=256, n_mel = 142
    spectrogram = np.absolute(spectrogram)
    if spec == 'mel':
        spectrogram = librosa.power_to_db(spectrogram) 
    else:
        spectrogram = librosa.amplitude_to_db(spectrogram)       
    return spectrogram

def data_preprocess(spec): #creat an array which store all the input data
    file_list = [ele for ele in os.listdir('./data/train') if ele[-3:]=='wav'] 
    train_data = list()
    for file in file_list:
        data = preprocess(file,spec=spec)
        train_data.append(data)
    train_data = np.array(train_data,dtype='float32')
    train_data.resize((train_data.shape[0],train_data.shape[1],train_data.shape[2],1)) #reshape the array with correct shape
    return train_data


def label_preprocess(): #create label for all train data
    file_list = [ele for ele in os.listdir('./data/train') if ele[-3:]=='wav'] 
    train_label = list(map(generate_label,file_list))
    train_label = np.array(train_label,dtype='int8')
    return train_label

def generate_label(file_name): #creating label for each input data
    label = np.zeros(3,dtype='int8') #create array with element where element no. = class no.
    ID = int(file_name.split('.')[0].split('_')[-1]) 
    label[ID] = 1
    return label



