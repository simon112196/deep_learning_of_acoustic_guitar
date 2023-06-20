import os
import numpy as np
from scipy.io import wavfile
from librosa import stft, cqt
from matplotlib import pyplot as plt
import librosa
from sklearn.metrics import roc_curve, auc, confusion_matrix
import librosa.display
from draw_tool import draw_spec

def preprocess(file_name, spec = 'mel', length=230000,): #transforming input data from waveform to spectrogram
    f, w_data = wavfile.read('./data/graph/%s'%file_name)
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

#drawing graph of spectrogram 
file1 = "fullPick_Dstring_A#3Martin.wav.wav" #target audio file 
# file2 = ""
title1 = 'Dstring_A#3Martin' #title of the spectrogram
title2 = 'Dstring_A4Sitar '
# files ={file1: title1, file2: title2}
# for file in files.keys():
#     stft = preprocess(file, 'stft')
#     title = files[file]
#     draw(stft, title)


stft = preprocess(file1, 'stft')
draw_spec(stft, title1, 'fft', 'stft')
# cqt2 = preprocess(file2, 'cqt')
cqt = preprocess(file1, 'cqt')


mel = preprocess(file1, 'mel')
# mel2 = preprocess(file2, 'mel')


draw_spec(cqt, title1, 'cqt_hz', 'cqt')
# draw(cqt2, title2, 'cqt_hz','cqt')
draw_spec(mel, title1, 'mel', 'mel')
# draw_spec(mel2, title2, 'mel', 'mel')




