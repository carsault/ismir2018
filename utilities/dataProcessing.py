#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 16:09:30 2018

@author: carsault
"""
#%% AUGMENTATION
import os
import numpy as np
import muda

def root(x):
    '''
    Keep only the root file name without extension

    Parameters
    ----------
    x: str
        file name with extension

    Returns
    -------
    str
        The name file without extension
    '''
    return os.path.splitext(os.path.basename(x))[0]

def augment(afile, jfile, deformer, outpath):
    jam = muda.load_jam_audio(jfile, afile, strict = False)
    base = root(afile)
    outfile = os.path.join(outpath, base) 
    for i, jam_out in enumerate(deformer.transform(jam)):
        muda.save('{}.{}.flac'.format(outfile, i),
                  '{}.{}.jams'.format(outfile, i),
                 jam_out, strict=False)
        
#%% DATA PUMP
def convert(aud, jam, pump, outdir):
    data = pump.transform(aud, jam)
    fname = os.path.extsep.join([root(aud), 'npz'])
    np.savez(os.path.join(outdir, fname), **data)
    
#%% Data import and format fitting
import json
import random as rd

def manual(tracks, audioSet, pitch = 6, random = False):
    '''
    Convert a set of tracks from .npz and .jams to our data format

    Parameters
    ----------
    tracks : Series
        list of the tracks to convert
    audioSet: class audioSet()
        object that contains our dataset
    pitch : int
        pitch to selct if random == False, without depitching if pitch = 6
    random : boolean
        randomize the pitch while converting or not
        
    Returns
    -------
    audioSet: class audioSet()
        object that contains our dataset
    '''
    for track in tracks.index:
        if random ==  True:
            #print(pitch)
            u = rd.randint(0,12)
        else:
            u = pitch
        if u == 12:
            fname = os.path.join( 'working/chords/pump', os.path.extsep.join([track , 'npz']))
            #fname = os.path.join( 'working/chords/pumpRWC', os.path.extsep.join([track , 'npz']))
        else :
            fname = os.path.join( 'working/chords/pump', os.path.extsep.join([track +"."+ str(u) , 'npz']))
            #fname = os.path.join( 'working/chords/pumpRWC', os.path.extsep.join([track +"."+ str(u) , 'npz']))
        data = np.load(fname)
        d2 = dict(data)
        data.close()
        data = d2
        data['cqt/mag'] = data['cqt/mag'][0]
        audioSet.data.append(data['cqt/mag'])
        if u == 12:
            fname = json.load(open('dataset/isophonics/metadataTest/Beatles/'+ track +".jams"))
            #fname = json.load(open('dataset/rwcpop/metadata/jam/'+ track +".jams"))
        else :
            fname = json.load(open('working/chords/augmentation/'+ track +"." + str(u) + ".jams"))
            #fname = json.load(open('working/chords/augmentationRWC/'+ track +"." + str(u) + ".jams"))
        acc = {}
        acc['labels'] = []
        acc['timeStart'] = []
        acc['timeEnd'] = []
        acc['key'] = []
        u = fname
        k = 0
        for nbacc in range(len(u['annotations'][0]['data'])):
            timeChordEnd = (u['annotations'][0]['data'][nbacc]["time"]+u['annotations'][0]['data'][nbacc]["duration"])
            timeKeyEnd = (u['annotations'][1]['data'][k]["time"]+u['annotations'][1]['data'][k]["duration"])
            acc['labels'].append(u['annotations'][0]['data'][nbacc]["value"])
            acc['timeStart'].append(u['annotations'][0]['data'][nbacc]["time"])
            acc['timeEnd'].append(timeChordEnd)
            acc['key'].append(u['annotations'][1]['data'][k]["value"])
            if timeKeyEnd < timeChordEnd and k + 1 < len(u['annotations'][1]['data']):
                k = k+1
        audioSet.metadata['chord'].append(acc)
    return audioSet

def manualOne(track, audioSet, pitch = 6): #not used and not up-to-date !!!!!!!!!!!
    '''
    Convert one track from .npz and .jams to our data format

    Parameters
    ----------
    tracks : str
        name of the track
    audioSet: class audioSet()
        object that contains our dataset
    pitch : int
        pitch to selct if random == False, without depitching if pitch = 6
        
    Returns
    -------
    audioSet: class audioSet()
        object that contains our dataset
    '''
    #for track in tracks.index:
        #fname = os.path.join( userdir + 'ismir2017_chords/working/chords/pump', os.path.extsep.join([track , 'npz']))
    fname = os.path.join('working/chords/pump', os.path.extsep.join([track +"."+ str(pitch) , 'npz']))
    data = np.load(fname)
    d2 = dict(data)
    data.close()
    data = d2
    data['cqt/mag'] = data['cqt/mag'][0]
    audioSet.data.append(data['cqt/mag'])
    fname = json.load(open('working/chords/augmentation/'+ track +"." + str(pitch) + ".jams"))
    #fname = json.load(open(userdir + 'ismir2017_chords/dataset/isophonics/metadata/Beatles/'+ track +".jams"))
    acc = {}
    acc['labels'] = []
    acc['timeStart'] = []
    acc['timeEnd'] = []
    u = fname
    for nbacc in range(len(u['annotations'][0]['data'])):
        acc['labels'].append(u['annotations'][0]['data'][nbacc]["value"])
        acc['timeStart'].append(u['annotations'][0]['data'][nbacc]["time"])
        acc['timeEnd'].append(u['annotations'][0]['data'][nbacc]["time"]+u['annotations'][0]['data'][nbacc]["duration"])
    audioSet.metadata['chord'].append(acc)
    return audioSet

def convMetaLBC(audioSet,transformOptions):
    '''
    Transform an audioSet to fit a CNN with a contexWindows

    Parameters
    ----------
    audioSet: class audioSet()
        object that contains our dataset
    transformOptions : dict
        information for the transformation
        
    Returns
    -------
    audioSet: class audioSet()
        audioSet.data is now by temporal frame with transformOptions["contextWindows"]
        audioSet.metadata['listBeatChord'] contains metadata for each temporal frame
    '''
    #convInput = []
    listBeatChord = [];
    listBeatPitchVect = [];
    listBeatPitchBass = [];
    listBeatKey = [];
    hopSizeS = (transformOptions["hopSize"] / transformOptions["resampleTo"]);
    nbData = 0;
    curData = 0;
    audioSet.metadata['listBeatChord'] = {};
    audioSet.metadata['pitchVector'] = {};
    audioSet.metadata['bass'] = {};
    audioSet.metadata['key'] = {};
#Count the number of frames
    for k in range(len(audioSet.data)):
        nbData = nbData + len(audioSet.data[k]) - transformOptions["contextWindows"] + 1
#Pre-allocate the windowed dataset
        #local finalData = options.modelType == 'ladder' and torch.Tensor(nbData, nbBands * options.contextWindows) or torch.Tensor(nbData, nbBands, options.contextWindows);
        #finalData = np.array(nbData, nbBands, options.contextWindows)
    finalData = {}
    #finalLabels = {};
#-- Parse the whole set of windows
    for k in range(len(audioSet.data)):
        #print(k)
        maxFrame = len(audioSet.data[k])
        for numFrame in range(maxFrame - transformOptions["contextWindows"]  + 1):
            nbrAcc = 0
            while numFrame + (transformOptions["contextWindows"]  / 2) + 0.5 > (audioSet.metadata['chord'][k]['timeEnd'][nbrAcc] / hopSizeS) and nbrAcc+1 < len(audioSet.metadata['chord'][k]['timeStart']):
                nbrAcc = nbrAcc+1;            
            #finalLabels[curData] = list(audioSet.metadata['chord'][k][0]['labels'][nbrAcc]);
            finalData[curData] = audioSet.data[k][range(numFrame, numFrame + transformOptions["contextWindows"])];
            #finalData[curData] = (finalData[curData] - finalData[curData].mean()) / finalData[curData].max();
            listBeatChord.append(audioSet.metadata['chord'][k]['labels'][nbrAcc]);
            listBeatPitchVect.append(audioSet.metadata['chord'][k]['labels'][nbrAcc]);
            listBeatPitchBass.append(audioSet.metadata['chord'][k]['labels'][nbrAcc]);
            listBeatKey.append(audioSet.metadata['chord'][k]['key'][nbrAcc]);
            curData = curData + 1;
#audioSet.data[k] = convInputTens;
        audioSet.metadata['listBeatChord'][k] = listBeatChord;
        audioSet.metadata['pitchVector'][k] = listBeatPitchVect;
        audioSet.metadata['bass'][k] = listBeatPitchBass;
        audioSet.metadata['key'][k] = listBeatKey;
        listBeatChord = []
        listBeatPitchVect = []
        listBeatPitchBass = []
        listBeatKey = []
    audioSet.data = finalData;
    #audioSet.metadata['listBeatChord'] = finalLabels
    return audioSet

from utilities.chordUtil import reduChord
from utilities.Chords2Vec_fun import mir_label_to_bin_chroma_vec, parse_mir_label_root, parse_mir_label, delta_root, normalized_note
import keras
def importAndTransf(audioSet, idx, pitch, transformOptions, alpha, modelType, dictChord, dictBass, single, random):
    '''
    Meta fonction that uses Manual() and convMetaLBC() then reduce to the alphabet, reshape depending on the model and compute the class_weight

    Parameters
    ----------
    audioSet: class audioSet()
        object that contains our dataset
    idx : Series
        list of the tracks to convert
    pitch : int
        pitch for the import, useless if random == True
    transformOptions : dict
        information for the transformation
    alpha : str
        name of the alphabet
    modelType : str
        name of the model, not the same out put if MLP or CNN
    dictChord : dict
        contains the chord alphabet elements
    dictBass : dict
        contains the bass alphabet elements
    single : bool
        applied on one or many tracks ?
    random : bool
        randomize the pitch during the import
    Returns
    -------
    x_full : array float32
        data to fit the  model
    x_full : array float32
        associated labels to fit the  model 
    class_weight : array float64
        weight of each class in the dataset
    '''
    # Get data and metadata
    if single == True:
        audioSet = manualOne(idx, audioSet, pitch, random)
    else:
        audioSet = manual(idx, audioSet, pitch, random = random)
    # Transform metadata
    nbrFile = len(audioSet.data)
    audioSet = convMetaLBC(audioSet,transformOptions)
    # Alphabet reduction
    for k in range(nbrFile):
        for j in range(len(audioSet.metadata['listBeatChord'][k])):
            chord, bass = parse_mir_label_root(audioSet.metadata['pitchVector'][k][j])
            audioSet.metadata['pitchVector'][k][j] = mir_label_to_bin_chroma_vec(reduChord(audioSet.metadata['listBeatChord'][k][j], 'reduceWOmodif'))
            audioSet.metadata['bass'][k][j] = bass
            audioSet.metadata['listBeatChord'][k][j] = reduChord(audioSet.metadata['listBeatChord'][k][j], alpha)
    x_fullList = []
    y_fullList = []
    pVector_fullList = []
    bass_fullList = []
    root_fullList = []
    key_fullList = []
    for k in range(nbrFile):
        for j in range(len(audioSet.metadata['listBeatChord'][k])):
            root, qual = parse_mir_label(audioSet.metadata['listBeatChord'][k][j])
            y_fullList.append(dictChord[audioSet.metadata['listBeatChord'][k][j]])
            pVector_fullList.append(audioSet.metadata['pitchVector'][k][j])
            #Calculate bass pitch
            if dictBass[audioSet.metadata['bass'][k][j]] == 12 :
                bass_fullList.append(12)
            else :
                if root == 'N' :
                    bass_fullList.append((dictBass[audioSet.metadata['bass'][k][j]]))
                else :
                    bass_fullList.append((dictBass[audioSet.metadata['bass'][k][j]]+delta_root('C',normalized_note(root)))%12)
            #Calculate root pitch
            if root == 'N' :
                root_fullList.append(12)
            else :
                root_fullList.append(delta_root('C',normalized_note(root))%12)
            #Calculate key
            key_fullList.append(audioSet.metadata['key'][k][j])
            
    for k in range(len(audioSet.data)):
            x_fullList.append(audioSet.data[k])
   
    row = len(x_fullList[0])
    col = len(x_fullList[0][0])
    
    x_full = np.asarray(x_fullList)
    if modelType == "mlp" or modelType == "ladder" or modelType == "mlp2" or  modelType == "mlpDusap":
        x_full = x_full.reshape(len(audioSet.data),row*col)
    else:
        x_full = x_full.reshape(len(audioSet.data),row,col,1)
    y_full = np.asarray(y_fullList)
    pVector_full = np.array(pVector_fullList)
    bass_full = np.array(bass_fullList)
    root_full = np.array(root_fullList)
    key_full = np.array(key_fullList)
    
    from sklearn.utils import class_weight
    class_weight = class_weight.compute_class_weight('balanced', np.unique(y_full), y_full)
    y_full = keras.utils.to_categorical(y_full, len(dictChord)).astype('float32')
    bass_full = keras.utils.to_categorical(bass_full, 13).astype('float32')
    root_full = keras.utils.to_categorical(root_full, 13).astype('float32')
    x_fullList = []
    y_fullList = []
    pVector_fullList = []
    bass_fullList = []
    root_fullList = []
    key_fullList = []
    
    return x_full, y_full, pVector_full, bass_full, root_full, key_full, class_weight