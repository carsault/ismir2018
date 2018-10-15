#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:46:32 2018

@author: carsault
"""
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, GaussianDropout, GaussianNoise, Activation, Lambda, Conv2D, MaxPooling2D, Input, Concatenate, Bidirectional, GRU, TimeDistributed


def convGru(input_shape, num_classes):
    x = Input(shape=input_shape, name = 'inputs') 
    b = BatchNormalization()(x)
    c0 = Conv2D(1, (5, 5), padding='same',
                                activation='relu',
                                data_format='channels_last')(b)
    c1 = Conv2D(36, (1, int(c0.shape[2])), padding='valid', activation='relu',
                                data_format='channels_last')(c0)
    r1 = Lambda(lambda x: keras.backend.squeeze(x, axis=2))(c1)
    rs = Bidirectional(GRU(256, return_sequences=False))(r1)
    p0 = Dense(num_classes, activation='softmax',bias_regularizer=keras.regularizers.l2())
    p1 = TimeDistributed(p0)(rs)
    model = Model(x, p1)
    model.summary()
    return model


def conv3article(input_shape, num_classes):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(GaussianNoise(0.3))
    model.add(Conv2D(16, kernel_size=(6, 25), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Dropout(0.5))
    model.add(Conv2D(20, kernel_size=(6, 27), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Conv2D(24, kernel_size=(6, 27), padding="same", activation='tanh'))
    model.add(Dropout(0.5))
    #model.add(TimeDistributed(Dense(1, activation='relu')))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model

def conv3articleUp(input_shape, num_classes):
    inputs = Input(shape=input_shape, name = 'inputs')    
    b = BatchNormalization(input_shape=input_shape)(inputs)
    g = GaussianNoise(0.3)(b)
    c1 = Conv2D(16, kernel_size=(6, 25), activation='tanh')(g)
    mp1 = MaxPooling2D(pool_size=(1, 3))(c1)
    d1 = Dropout(0.5)(mp1)
    c2 = Conv2D(20, kernel_size=(6, 27), activation='tanh')(d1)
    d2 = Dropout(0.5)(c2)
    c3 = Conv2D(24, kernel_size=(6, 27), padding="same", activation='tanh')(d2)
    d3 = Dropout(0.5)(c3)
    f = Flatten()(d3)
    f = Dense(200, activation='relu')(f)
    d4 = Dropout(0.5)(f)
    pV = Dense(12, name = 'pV', activation='softmax')(d4)
    bass = Dense(13, name = 'bass', activation='softmax')(d4)
    root = Dense(13, name = 'root', activation='softmax')(d4)
    conc = Concatenate()([d4, pV, bass, root])
    out = Dense(num_classes, name = 'out', activation='softmax')(conc)
    model = Model(inputs=inputs, outputs=[out, pV, bass, root])
    model.summary()
    return model