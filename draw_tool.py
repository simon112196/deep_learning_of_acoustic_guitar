import os
import numpy as np
from matplotlib import pyplot as plt
import librosa
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import librosa.display


def draw_spec(spec, title, yax): #Draw spectrogram
    plt.figure(1)
    librosa.display.specshow(spec,
                            y_axis=yax,)
    plt.title(title)
    plt.colorbar(format='%+2.0f')
    fname = os.path.join(".", "%r_%r"%(title, spec))
    plt.savefig(fname)
    plt.show()
    
lw = 2 #line width
n_classes = 3 #number of class
#    
# Code for drawing roc curve
# This code was adapted from DLology published on Apr 2018
# accessed 15-7-2022
# https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
# Size of the graph was modified, code is add for automatic saving
# 
def roc_draw(x_test, y_test, model): #Draw Roc Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = model.predict(x_test)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(1)
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['red', 'blue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    fname = os.path.join(".", "roc")
    plt.savefig(fname)
    plt.close(1)

def loss_accuracy_draw(history, fold_no=1): #Draw Loss and Accuracy curve
    plt.figure(2)
    plt.title('Loss')
    plt.plot(history.history['loss'], label = 'Loss')
    plt.plot(history.history['val_loss'], label = 'Val Loss')
    plt.legend()
    fname = os.path.join(".", "loss%s"%fold_no)
    plt.savefig(fname)
    plt.close(2)
    plt.figure(3)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label = 'Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
    plt.legend()
    fname = os.path.join(".", "accuracy%s"%fold_no)
    plt.savefig(fname)
    plt.close(3)


def plot_graph(his_acc, title): #Draw graph in gridsearch
    plt.figure(figsize=(15, 8))
    for acc in his_acc.keys():
        plt.plot(his_acc[acc], 
                 label=acc,
                    linewidth=3)
        
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()    
