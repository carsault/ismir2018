#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:29:20 2018

@author: carsault
"""

#%%
from keras.models import load_model
'''
try:
    from matplotlib import pyplot as plt
except:
    matplotlib.use('agg')'''
import pickle
import pandas as pd
#Construct tonnetz matrix
import utilities.distance as distances
from utilities.training import wrap_loss_function
from keras import backend as K
import numpy as np

def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])

loss = 'tonnetz'
'''
with open('model_conv3article_a0_categorical_crossentropy_history.p', 'rb') as pickle_file:
    history = pickle.load(pickle_file)

with open('model_conv3article_a0_categorical_crossentropy_idx_test.p', 'rb') as pickle_file:
    idx_test = pd.read_pickle(pickle_file)'''

def loadModelData(modelName, loss, seed):
    seed = str(seed)
    dictChord = {}
    with open('modelSave' + seed +'/' + modelName + '/' + modelName + '_history.p', 'rb') as pickle_file:
        history = pickle.load(pickle_file)
    with open('modelSave' + seed +'/' + modelName + '/' + modelName + '_listChord.p', 'rb') as pickle_file:
        listChord = pickle.load(pickle_file)
    with open('modelSave' + seed +'/' + modelName + '/' + modelName + '_idx_test.p', 'rb') as pickle_file:
        idx_test = pd.read_pickle(pickle_file)
    for i in range(len(listChord)):
        dictChord[listChord[i]] = i
        
    if loss == 'categorical_crossentropy' or loss == 'hinge':
        model = load_model('modelSave' + seed +'/' + modelName + '/' + modelName + '.hdf5')
    elif loss == 'tonnetz':
        tf_mappingR = distances.tonnetz_matrix((invert_dict(dictChord),invert_dict(dictChord)))
        tf_mappingR = (tf_mappingR + np.mean(tf_mappingR))
        tf_mappingR = 1./ tf_mappingR
        tf_mappingR = (tf_mappingR) / np.max(tf_mappingR)
        tf_mapping = K.constant(tf_mappingR)
        loss=wrap_loss_function(tf_mapping = tf_mapping)
        model = load_model('modelSave' + seed +'/'+ modelName + '/' + modelName + '.hdf5', custom_objects={'loss_function': loss})
    elif loss == 'euclidian':
        tf_mappingR = distances.euclid_matrix((invert_dict(dictChord),invert_dict(dictChord)))
        tf_mappingR = (tf_mappingR + np.mean(tf_mappingR))
        tf_mappingR = 1./ tf_mappingR
        tf_mappingR = (tf_mappingR) / np.max(tf_mappingR)
        tf_mapping = K.constant(tf_mappingR)
        loss=wrap_loss_function(tf_mapping = tf_mapping) 
        model = load_model('modelSave' + seed +'/' + modelName + '/' + modelName + '.hdf5', custom_objects={'loss_function': loss})
    elif loss == 'categorical_hinge':
        tf_mappingR = np.identity(len(listChord))
        tf_mapping = K.constant(tf_mappingR)
        loss=wrap_loss_function(tf_mapping = tf_mapping)
        model = load_model('modelSave' + seed +'/'+ modelName + '/' + modelName + '.hdf5', custom_objects={'loss_function': loss})
    return model, history, listChord, idx_test

def plotHistory(history):
    history = pd.DataFrame.from_dict(history)
    
    #plot accuracy
    plt.figure()
    plt.plot(history['out_acc'], label='Training accuracy')
    plt.plot(history['val_out_acc'], label='Validation accuracy')
    plt.legend(loc='best')
    
    #plot loss function
    plt.figure()
    plt.plot(history['loss'], label='Training accuracy')
    plt.plot(history['val_loss'], label='Validation accuracy')
    plt.legend(loc='best')
    
import pandas as pd
import jams
from tqdm import tqdm_notebook as tqdm
import json
from utilities.chordUtil import reduChord
import numpy as np
import librosa
import mir_eval
from Analyse_ISMIR.ACEAnalyzer import *
from utilities.distance import *

#alpha2 = 'a0'
sr = 44100
hop_length = 4096
transformOptions = {}
transformOptions["contextWindows"] = 15    
transformOptions["hopSize"] = hop_length   
transformOptions["resampleTo"] = sr


def score_model(pump, model, alpha, idx, listChord,
                features= 'working/chords/pump/',
                refs= 'dataset/isophonics/metadataTest/Beatles'):
    
    countTonnezt = {}
    countEuclid = {}
    #audioSetTest = audioSet()
    #pitch=12
    #(x_test, y_test, pV_test, bass_test, root_test, key_test, class_weight) = importAndTransf(audioSetTest, idx, pitch, transformOptions, alpha, modelType, dictChord, dictBass, single = False, random = False) 
    results = {}
    totalFrame = 0
    Analyzer = ACEAnalyzer()
    hopSizeS = (transformOptions["hopSize"] / transformOptions["resampleTo"])
    
    for item in tqdm(idx.index):
        #jam = jams.load('{}/{}.jams'.format(refs, item), validate=False)
        datum = np.load('{}/{}.npz'.format(features, item))['cqt/mag']
        print(item)

        #confidence = 1
        interv = np.ndarray(shape=(1,2))
        interv[0][0] = 0
        interv[0][1] = 1
        
        #load and transform jam
        fname = json.load(open('dataset/isophonics/metadataTest/Beatles/'+ item + ".jams"))
        
        u = fname
        '''
        for nbacc in range(len(u['annotations'][0]['data'])):
            t_start = u['annotations'][0]['data'][nbacc]["time"]
            t_end = u['annotations'][0]['data'][nbacc]["time"]+u['annotations'][0]['data'][nbacc]["duration"]
            #vd = reduChord(reduChord(u['annotations'][0]['data'][nbacc]["value"], alpha2),'aMirex')
            vd = reduChord(u['annotations'][0]['data'][nbacc]["value"], 'reduceWOmodif')
            ann_true.append(time=t_start,
                               duration=t_end-t_start,
                               value=vd,
                               confidence=float(confidence))
        '''
        maxFrame = len(datum[0])
        
        for numFrame in range(maxFrame - transformOptions["contextWindows"]  + 1):
            #ann = jams.Annotation('chord')
            #ann_true = jams.Annotation('chord')
            #select frame in sample
            nbrAcc = 0
            nbrKey = 0
            dist = 1
            while numFrame + (transformOptions["contextWindows"]  / 2) + 0.5 > (u['annotations'][0]['data'][nbrAcc]["time"]+u['annotations'][0]['data'][nbrAcc]["duration"] )/ hopSizeS and nbrAcc+1 < len(u['annotations'][0]['data']):
                nbrAcc = nbrAcc+1; 
                       
            while numFrame + (transformOptions["contextWindows"]  / 2) + 0.5 > (u['annotations'][1]['data'][nbrKey]["time"]+u['annotations'][1]['data'][nbrKey]["duration"] )/ hopSizeS and nbrAcc+1 < len(u['annotations'][1]['data']):
                nbrKey = nbrKey+1; 
            start = numFrame + (transformOptions["contextWindows"]  + 1)/2
            #get in time
            t_start, t_end = librosa.core.frames_to_time([start, start+1],
                                                sr=sr,
                                                hop_length=hop_length)

            #get predicted value
            #vd = reduChord(listChord[np.argmax(model.predict(datum[:,numFrame:numFrame+15]))],'aMirex')
            true = reduChord(u['annotations'][0]['data'][nbrAcc]["value"], 'reduceWOmodif')
            key = u['annotations'][1]['data'][nbrKey]["value"]
            pred = listChord[np.argmax(model.predict(datum[:,numFrame:numFrame+15])[0])]
        
            '''ann.append(time=t_start,
                               duration=t_end-t_start,
                               value=vd,
                               confidence=float(confidence))'''
            
            results[totalFrame] = mir_eval.chord.evaluate(interv,[true], interv, [pred])
            results[totalFrame].update({'DistTonnezt' : distance_tonnetz(true, pred)})
            results[totalFrame].update({'DistEuclid' : distance_euclid(true, pred)})
            
            true = reduChord(u['annotations'][0]['data'][nbrAcc]["value"], alpha)
            if true == pred:
                dist = 0
            else:
                dist = 1
                
            results[totalFrame].update({'DistCateg' : dist})
            
            Analyzer.compare(chord = true, target = pred, key = key, base_alpha = a5, print_comparison = False)
            totalFrame = totalFrame + 1

        #results[item] = jams.eval.chord(jam.annotations['chord', 0], ann)
        #results[item] = jams.eval.chord(ann_true, ann)

        #print(jam.annotations['chord', 0])
        
    return pd.DataFrame.from_dict(results, orient='index'), Analyzer, pd.DataFrame.from_dict(countTonnezt, orient='index'), pd.DataFrame.from_dict(countEuclid, orient='index')

def plotAcc(df):
    dfr = df[['thirds', 'triads', 'tetrads', 'root', 'mirex', 'majmin', 'sevenths']]
    print(dfr.describe())
    plt.figure()
    dfr.boxplot();
    print(dfr.describe().loc['mean'])