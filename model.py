#!/opt/anaconda3/bin/python

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Input, 
    Dense, 
    Conv2D, 
    Flatten,
    Dropout,
)

L2_RATE=0.0001

def KerasModel(shape, drop_rate, f1, f2, f3, isCompile=True):

    In = Input(shape=shape)

    x = Conv2D(f1, (7, 7), activation='relu', strides=(3,3), kernel_regularizer=l2(L2_RATE), padding='same')(In)

    x = Conv2D(f2, (7, 7), activation='relu', strides=(3,3),  kernel_regularizer=l2(L2_RATE), padding='same')(x) 
    x = Dropout(drop_rate)(x)  
    x = Conv2D(f3, (3, 3), activation='relu', strides=(2,2), kernel_regularizer=l2(L2_RATE), padding='same')(x)    
    x = Dropout(drop_rate)(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=In, outputs=x)
    opt = Adam(1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    KerasModel()
