#!/opt/anaconda3/bin/python

import librosa
from model import KerasModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split, KFold
from librosa import stft, cqt
import librosa
from sklearn.metrics import confusion_matrix
import warnings
from draw_tool import roc_draw, loss_accuracy_draw, plot_graph
from preprocess import preprocess, data_preprocess, generate_label, label_preprocess

warnings.filterwarnings('ignore')

# sr = 44100
# hop_length = 512
# n_fft = 2048
# win_length = 2048

class MyCallback(Callback):
    def __init__(self):
        pass
    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights('./models/save_%d.h5' % epoch)


def n_fold(train_data,train_label,name, para, f1, f2, f3, batch, epoch): #5 fold cross validation
    seed = 7
    tf.random.set_seed(seed)
    num_folds = 5
    fold_no = 1
    kfold = KFold(n_splits=num_folds, shuffle=True)
    acc_per_fold = []
    loss_per_fold = []
    for train_index, test_index in kfold.split(train_data, train_label):
        X_train, X_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_label[train_index], train_label[test_index]
        shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        model = KerasModel(shape, para, f1=f1, f2=f2, f3=f3)
        cb = MyCallback()
        callbacks=[cb,EarlyStopping(monitor='loss', patience=4, verbose=1,mode='auto')]
        history = model.fit(X_train, y_train,batch_size=batch,epochs=epoch,verbose=1,validation_split=0.1,callbacks=callbacks)
        result = model.evaluate(X_test, y_test, batch_size= 100)
        print('test loss, test acc:', result)
        # print("loss, acc: %r with %s : %s" %(result,name ,para))
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {result[0]}; {model.metrics_names[1]} of {result[1]*100}%')
        acc_per_fold.append(result[1] * 100)
        loss_per_fold.append(result[0])
        fold_no += 1
        loss_accuracy_draw(history, fold_no)
        roc_draw(X_test, y_test, model)
        # == Provide average scores ==
    print('Filter(%r, %r, %r)'%(f1,f2,f3))
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('Average scores:')
        print(f'> Accuracy: {np.mean(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
#    
# Using as the skeleton of the system
# This code was adapted from Github published on 19 Apr 2019
# accessed 15-7-2022
# https://github.com/kimmo1019/audioNet
# Added code to spliting data and drawing graphs 
# 
def train(train_data,train_label,name, para, f1, f2, f3, epoch, batch, sp=-1): #train the model once
    seed = 7
    tf.random.set_seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(train_data,train_label, test_size=0.2, random_state=42)
    shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = KerasModel(shape, para, f1=f1, f2=f2, f3=f3)
    cb = MyCallback()
    callbacks=[cb,EarlyStopping(monitor='loss', patience=4, verbose=1,mode='auto')]
    history = model.fit(X_train, y_train,batch_size=batch,epochs=epoch,verbose=1,validation_split=0.1,callbacks=callbacks)
    result = model.evaluate(X_test, y_test, batch_size= 100) 
    print('Filter(%r, %r, %r)'%(f1,f2,f3))
    print('test loss, test acc:', result)
    # print("loss, acc: %r with %s : %s" %(result,name ,para))
    roc_draw(X_test, y_test, model)
    loss_accuracy_draw(history)   
    
        
    pred = model.predict(X_test) #prediction result
    pred = np.argmax (pred, axis = 1)
    y_test=np.argmax(y_test, axis=1)
    print(confusion_matrix(y_test, pred))
    return(history)
    


batch_size = [32, 64, 128]
epochs = [30,50, 70,100]
n_ffts = [2048, 3072, 4096]
hop_lengths = [128, 256, 512, 1024, 2048]
n_mels = [128, 142 ,256]
filters = [16,32]


if __name__ == '__main__':
    spec = 'mel'
    train_data = data_preprocess(spec = spec)
    train_label = label_preprocess()
    acc_measure = {}
    acc_arr = {}
    f1=16
    f2=32
    f3=32
    epoch=45
    batch=64
    if spec == 'mel':
        batch=32
    elif spec == 'stft':
        f1=32
        f2=16
        f3=16
        epoch=30

    # for f1 in filters:
    #     for f2 in filters:
    #         for f3 in filters:
    # for batch in batch_size:
    
    arr = n_fold(train_data, train_label, 'drop rate', 0.3, f1=f1, f2=f2, f3=f3, epoch = epoch, batch = batch) #stft(32,16,16): epoch=30, batch=64, cqt(16,32,32): epoch=45, batch=64, mel(16,32,32): epoch=45, batch=32
    
    # index = '(%d, %d, %d)'%(f1,f2,f3)
    # acc_arr[index] = arr.history["accuracy"] 
    # plot_graph(acc_arr, "Layer Configuration Test")
        
        
    
   
    
    
