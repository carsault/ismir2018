#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:08:34 2018

@author: Tristan
"""
augmentAudio = True  #Augment data with pitch shifting
processTransf = True #Do the transformation with pump

lr = 0.00002
#lr = 0.01
batch_size = 500
epochs = 1500
#%%option parser
opt = ['conv3article', 'a0', 'categorical_crossentropy', 20180323]

from optparse import OptionParser
usage = "useage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("-o", "--all", type="string", dest="search_and", help="options")
(options, args) = parser.parse_args()
if options.search_and is not None:
    opt = options.search_and.split(',')
modelType = opt[0]
alpha = opt[1]
loss = opt[2]
seed = int(opt[3])
name = 'model_' + opt[0] + '_' + opt[1] + '_' + opt[2]

#%% AUGMENTATION
from utilities.dataProcessing import *
from joblib import Parallel, delayed
import jams
import muda
import pickle
from utilities.chordUtil import *

AUDIO = jams.util.find_with_extension('dataset/isophonics/audio/Beatles', 'mp3')
ANNOS = jams.util.find_with_extension('dataset/isophonics/metadata/Beatles', 'jams') #jams file from https://github.com/marl/jams-data/tree/master/datasets

print(str(len(AUDIO)) + " songs")
# Make sure there are the same number of files
assert len(AUDIO) == len(ANNOS)


# And that they're in agreement
for (_1, _2) in zip(AUDIO, ANNOS):
    try :
        assert root(_1) == root(_2)
    except AssertionError as error:
        # Output expected AssertionErrors.
        print(root(_1))    
assert all([root(_1) == root(_2) for (_1, _2) in zip(AUDIO, ANNOS)])

OUTDIR = 'working/chords/augmentation/'

# Create the augmentation engine
pitcher = muda.deformers.PitchShift(n_semitones=[-1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6])

i = 0
with open('working/chords/muda.pkl', 'wb') as fd:
    pickle.dump(pitcher, fd)
    
if augmentAudio == True:
    for (aud, jam) in zip(AUDIO, ANNOS):
        print(i)
        i = i+1
        augment(aud, jam, pitcher, OUTDIR)

#%% DATA PUMP
import pumppExtra as pumpp
import pandas as pd
from tqdm import tqdm_notebook as tqdm

# Build a pump
sr = 44100
hop_length = 4096

p_feature = pumpp.feature.CQTMag(name='cqt', sr=sr, hop_length=hop_length, log=True, conv='tf', n_octaves=6)
p_chord_tag = pumpp.task.ChordTagTransformer(name='chord_tag', sr=sr, hop_length=hop_length, sparse=True)
p_chord_struct = pumpp.task.ChordTransformer(name='chord_struct', sr=sr, hop_length=hop_length, sparse=True)

pump = pumpp.Pump(p_feature, p_chord_tag, p_chord_struct)

# Save the pump
with open('working/chords/pump.pkl', 'wb') as fd:
    pickle.dump(pump, fd)
    
OUTDIR = 'working/chords/pump/'
#%%
if processTransf == True:
    for (aud, jam) in zip(AUDIO, ANNOS):
        convert(aud, jam, pump, OUTDIR);
        
# Make the artist index
index = pd.Series()
null_artist = 0

# Augmented data
from glob import glob
AUDIO_A = sorted(glob('working/chords/augmentation/*.flac'))
ANNOS_A = sorted(glob('working/chords/augmentation/*.jams'))

# Make sure there are the same number of files
assert len(AUDIO_A) == len(ANNOS_A)
# And that they're in agreement
assert all([root(_1) == root(_2) for (_1, _2) in zip(AUDIO_A, ANNOS_A)])

if processTransf == True:
    for (aud, jam) in zip(AUDIO_A, ANNOS_A):
        print(aud)
        convert(aud, jam, pump, OUTDIR);
        
for ann in tqdm(ANNOS):
    J = jams.load(ann, validate=False)
    if not J.file_metadata.artist:
        artist = 'artist_{:05d}'.format(null_artist)
        null_artist += 1
    else:
        artist = J.file_metadata.artist
        
    index[root(ann)] = artist

index.to_json('working/chords/artist_index.json')

for ann in tqdm(ANNOS):
    J = jams.load(ann, validate=False)
    print('{}: {}'.format(root(ann), len(J.annotations['chord']))) 
    

#%%
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import ShuffleSplit
index = pd.read_json('working/chords/artist_index.json', typ='series')
#splitter_tt = ShuffleSplit(n_splits=1,test_size = 0.2,random_state=seed)
splitter_tt = ShuffleSplit(n_splits=1, test_size = 0.9,random_state=seed) #forLocalTest
print(splitter_tt.split(index))

for train_, test in splitter_tt.split(index, groups=list(index)):
    idx_train_ = index.iloc[train_]
    idx_test = index.iloc[test]
    #SEED = SEED + 1
    splitter_tv = ShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    for train, val in splitter_tv.split(idx_train_, groups=list(idx_train_)):
        idx_train = idx_train_.iloc[train]
        idx_val = idx_train_.iloc[val]
        
# Create the dataset
class audioSet:
    def __init__(self):
        self.data = []
        self.metadata = {}
        self.metadata['chord'] = []
 #   pass

#%% 
row = 0
col = 0
transformOptions = {}
transformOptions["contextWindows"] = 15    
transformOptions["hopSize"] = hop_length   
transformOptions["resampleTo"] = sr
from utilities.chordUtil import getDictChord
dictChord, listChord = getDictChord(eval(alpha))

from utilities.dataProcessing import importAndTransf

#%%
pitch = 12 #we can change on which pitch we do the validation (natural pitch == 12)
audioSetTest = audioSet()
(x_val, y_val, pV_val, bass_val, root_val, key_val, class_weight) = importAndTransf(audioSetTest, idx_val, pitch, transformOptions, alpha, modelType, dictChord, dictBass, single = False, random = False) 
audioSetTrain = audioSet()
pitch = 1 #we don't care if random == True in importAndTransf()

#%%
(x_train, y_train, pV_train, bass_train, root_train, key_train, class_weight) = importAndTransf(audioSetTrain, idx_train, pitch, transformOptions, alpha, modelType, dictChord, dictBass, single = False, random = True)

num_classes = len(listChord)

#Construct tonnetz matrix
import utilities.distance as distances
from utilities.training import wrap_loss_function
from keras import backend as K
import numpy as np

def invert_dict(d):
    return dict([(v, k) for k, v in d.items()])

if loss == 'categorical_crossentropy':
    loss=keras.losses.categorical_crossentropy
elif loss == 'mean_squared_error':
    loss=keras.losses.mean_squared_error
elif loss == 'hinge':
    loss=keras.losses.hinge
elif loss == 'categorical_hinge':
    tf_mappingR = np.identity(num_classes)
    tf_mapping = K.constant(tf_mappingR)
    loss=wrap_loss_function(tf_mapping = tf_mapping)
elif loss == 'tonnetz':
    tf_mappingR = distances.tonnetz_matrix((invert_dict(dictChord),invert_dict(dictChord)))
    tf_mappingR = (tf_mappingR + np.mean(tf_mappingR))
    tf_mappingR = 1./ tf_mappingR
    tf_mappingR = (tf_mappingR) / np.max(tf_mappingR)
    tf_mapping = K.constant(tf_mappingR)
    loss=wrap_loss_function(tf_mapping = tf_mapping)
elif loss == 'euclidian':
    tf_mappingR = distances.euclid_matrix((invert_dict(dictChord),invert_dict(dictChord)))
    tf_mappingR = (tf_mappingR + np.mean(tf_mappingR))
    tf_mappingR = 1./ tf_mappingR
    tf_mappingR = (tf_mappingR) / np.max(tf_mappingR)
    tf_mapping = K.constant(tf_mappingR)
    loss=wrap_loss_function(tf_mapping = tf_mapping)    
else:
    raise ValueError('Cost function named '+loss+' not defined')
    
import gc
from random import randint
class randomSelect(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        audioSetTrain = audioSet()
        pitch = 1 #we don't care if random == True in importAndTransf()
        x_train = []
        y_train = []
        gc.collect()
        gc.collect()
        (x_train, y_train, pV_train, bass_train, root_train, key_train, class_weight) = importAndTransf(audioSetTrain, idx_train, pitch, transformOptions, alpha, modelType, dictChord, dictBass, single = False, random = True)
    
#%%Select model
randomS = randomSelect()
input_shape = (len(x_train[0]), len(x_train[0][0]), 1, )
from utilities.models import *
import numpy as np
import os
import errno
import tensorflow as tf

#remove with guillimin
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "1"
set_session(tf.Session(config=config))

try:
    os.mkdir('modelSave' + str(seed))
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

try:
    os.mkdir('modelSave' + str(seed) + '/' + name)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

#%%
if modelType == "conv3article":
    model = conv3article(input_shape, num_classes)
    #keras.utils.plot_model(model, to_file='model1.png', show_shapes=True, show_layer_names=False, rankdir='TB')
    model.compile(loss=loss,
              optimizer=keras.optimizers.Adam(lr=lr),
              metrics=['categorical_accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, verbose = 1, class_weight = class_weight,
          epochs=epochs, validation_data=(x_val, y_val), callbacks=[randomS, keras.callbacks.ModelCheckpoint('modelSave'+ str(seed) + '/' + name + '/' + name + '.hdf5',
                                                               save_best_only=True,
                                                               verbose=1), keras.callbacks.EarlyStopping(verbose=1, patience=500), keras.callbacks.ReduceLROnPlateau(verbose=1, patience = 50)]).history
elif modelType == "conv3articleUp":
    model = conv3articleUp(input_shape, num_classes)
    model.compile(loss={'out' : loss, 'pV' : keras.losses.mean_squared_error, 'bass' : keras.losses.categorical_crossentropy, 'root' : keras.losses.categorical_crossentropy},
              optimizer=keras.optimizers.Adam(lr=lr),metrics=['categorical_accuracy'])
    history = model.fit(x_train, [y_train, pV_train, bass_train, root_train], batch_size=batch_size, verbose = 1, class_weight = [class_weight, np.ones(12), np.ones(13), np.ones(13)],
          epochs=epochs, validation_data=(x_val, [y_val, pV_val, bass_val, root_val]), callbacks=[randomS, keras.callbacks.ModelCheckpoint('modelSave/' + name + '/' + name + '.hdf5',
                                                               save_best_only=True,
                                                               verbose=1), keras.callbacks.EarlyStopping(verbose=1, patience=50), keras.callbacks.ReduceLROnPlateau(verbose=1)]).history
    
if modelType == "convGru":
    model = convGru(input_shape, num_classes)
    model.compile(loss=loss,
              optimizer=keras.optimizers.Adam(lr=lr),
              metrics=['categorical_accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, verbose = 1, class_weight = class_weight,
          epochs=epochs, validation_data=(x_val, y_val), callbacks=[randomS, keras.callbacks.ModelCheckpoint('modelSave/' + name + '/' + name + '.hdf5',
                                                               save_best_only=True,
                                                               verbose=1), keras.callbacks.EarlyStopping(verbose=1, patience=50), keras.callbacks.ReduceLROnPlateau(verbose=1)]).history
#%% Training procedure
pickle.dump(history, open('modelSave'+ str(seed) + '/' + name + '/' + name + '_history.p', "wb"))
pickle.dump(listChord, open('modelSave'+ str(seed) + '/' + name + '/' + name + '_listChord.p', "wb"))
pickle.dump(idx_test, open('modelSave'+ str(seed) + '/' + name + '/' + name + '_idx_test.p', "wb"))
pickle.dump(idx_val, open('modelSave'+ str(seed) + '/' + name + '/' + name + '_idx_val.p', "wb"))
pickle.dump(idx_train, open('modelSave'+ str(seed) + '/' + name + '/' + name + '_idx_train.p', "wb"))
#%%
#exit()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
