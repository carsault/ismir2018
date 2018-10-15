#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 16:00:13 2018

@author: carsault
"""

#%%
from utilities.test import loadModelData, plotHistory, plotAcc, score_model
import pickle
from utilities.test import *
import pandas as pd


from optparse import OptionParser
usage = "useage: %prog [options]"
parser = OptionParser(usage)
parser.add_option("-o", "--all", type="string", dest="search_and", help="options")
(options, args) = parser.parse_args()
if options.search_and is not None:
    opt = options.search_and.split(',')
seed = str(opt[0])


#configure GPU
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


# Reload the pump
with open('working/chords/pump.pkl', 'rb') as fd:
    pump = pickle.load(fd) 
#%% 
# Stats for one model 
'''print(history.keys())
modelName = "model_conv3article_a0_categorical_crossentropy"
model, history, listChord, idx_test = loadModelData(modelName)
plotHistory(history)
df = score_model(pump, model, idx_test)
plotAcc(df)'''
#%%
modelStatsStd = []
modelStatsMean = []
modelStatsMed = []
modelNames = []
#alpha = ['a5']
#loss = ['tonnetz']
modelStruct = ['conv3article']
#modelStruct = ['conv3article', 'conv3articleUp'];
alpha = ['a0', 'a2', 'a5'];
loss = ['categorical_crossentropy', 'tonnetz', 'euclidian', 'categorical_hinge', 'hinge'];
modelTag = ['convA', 'convB'];
lossTag = ['categori', 'tonnetz', 'euclidian', 'catHing', 'hinge'];
for m in range(len(modelStruct)):
    for a in range(len(alpha)):
        for l in range(len(loss)):
            modelName = "model_" + modelStruct[m] + "_" + alpha[a] + "_" + loss[l]
            print(modelName)
            model, history, listChord, idx_test = loadModelData(modelName, loss[l],seed)
            df, Analyzer, distTonnetz, distEuclid = score_model(pump, model, alpha[a], idx_test, listChord)
            modelStatsMean.append(df.describe().loc['mean'])
            #modelStatsMed.append(df.describe().loc['50%'])
            #modelStatsStd.append(df.describe().loc['std'])
            pickle.dump(Analyzer, open('modelSave' + seed +'/' + modelName + '/Analyzer.p', "wb"))
            modelNames.append(modelTag[m] + "_" + alpha[a] + "_" + lossTag[l])
            
alphaList = ['thirds', 'triads', 'tetrads', 'root', 'mirex', 'majmin', 'sevenths', 'DistTonnezt', 'DistEuclid', 'DistCateg']
res = []
for i in range(len(alphaList)):            
    tab = pd.concat(modelStatsMean, keys = modelNames)
    tab = pd.DataFrame(tab)
    tab = tab.swaplevel(i=-2, j=-1, axis=0)
    tab = tab.loc[alphaList[i]]
    tab.rename(columns = {'mean':alphaList[i]}, inplace = True)
    res.append(tab)
res = pd.concat(res, axis=1)
pickle.dump(res, open('modelSave' + seed +'/resultsMean.p', "wb"))


'''
res = []
for i in range(len(alphaList)):            
    tab = pd.concat(modelStatsMed, keys = modelNames)
    tab = pd.DataFrame(tab)
    tab = tab.swaplevel(i=-2, j=-1, axis=0)
    tab = tab.loc[alphaList[i]]
    tab.rename(columns = {'50%':alphaList[i]}, inplace = True)
    res.append(tab)
res = pd.concat(res, axis=1)
pickle.dump(res, open('modelSave/resultsMed.p', "wb"))

res = []
for i in range(len(alphaList)):            
    tab = pd.concat(modelStatsStd, keys = modelNames)
    tab = pd.DataFrame(tab)
    tab = tab.swaplevel(i=-2, j=-1, axis=0)
    tab = tab.loc[alphaList[i]]
    tab.rename(columns = {'std':alphaList[i]}, inplace = True)
    res.append(tab)
res = pd.concat(res, axis=1)
pickle.dump(res, open('modelSave/resultsStd.p', "wb"))'''
#%%
import pickle
import pandas as pd
res = {}
seed = ['201803230', '201803231', '201803232', '201803233', '201803234']
#seed = ['201803230', '201803231', '201803232', '201803233']
for s in range(len(seed)):
    with open('modelSave' + seed[s] + '/resultsMean.p', 'rb') as pickle_file:
        u = pickle.load(pickle_file)
        #res[s] = res[s][:9:3]
        res[s] = u[::5].rename(index={'convA_a0_categori' : 'A0_D0', 'convA_a2_categori' : 'A1_D0', 'convA_a5_categori' : 'A2_D0'})
        res[s] = pd.concat((res[s],u[1::5].rename(index={'convA_a0_tonnetz' : 'A0_D1', 'convA_a2_tonnetz' : 'A1_D1', 'convA_a5_tonnetz' : 'A2_D1'})))
        res[s] = pd.concat((res[s],u[2::5].rename(index={'convA_a0_euclidian' : 'A0_D2', 'convA_a2_euclidian' : 'A1_D2', 'convA_a5_euclidian' : 'A2_D2'})))
'''print(res4[:9:])
with open('modelSave3b/resultsMean.p', 'rb') as pickle_file:
    res3 = pickle.load(pickle_file)
    res3 = res3[:9:]'''
#print(res3[:9:])
df_concatter = pd.concat((res[0],res[1]))
df_concatbis = pd.concat((res[2],res[2]))
df_concat = pd.concat((df_concatbis,df_concatter))
df_concat.index.name = ''
fig = df_concat.boxplot(column='mirex',by = df_concat.index).get_figure()
fig.suptitle('')
#fig.xlabel('')
#fig = df_concat.boxplot(column='majmin',by = df_concat.index).get_figure()
#fig.title("Boxplot of Something")
fig.savefig('plots/tetrads.svg')
#%%
by_row_index = df_concat.groupby(df_concat.index)
#print(by_row_index.head())
print(by_row_index.std())
print(by_row_index.mean())
#df_concat.mean()
import matplotlib.pyplot as plt
by_row_index.boxplot(column='DistEuclid')
#%%
import pickle
from Analyse_ISMIR import ACEAnalyzer
alpha = ['a0', 'a2', 'a5'];
loss = ['categorical_crossentropy', 'tonnetz', 'euclidian', 'categorical_hinge', 'hinge'];
loss = ['categorical_crossentropy', 'tonnetz', 'euclidian'];
loss = ['euclidian'];
modelStruct = ['conv3article']
seed = ['201803230', '201803231', '201803232', '201803233', '201803234']
#seed = ['201803230', '201803231', '201803232', '201803233']
listAnalyzer = {}
for m in range(len(modelStruct)):
    for a in range(len(alpha)):
        for l in range(len(loss)):
            for s in range(len(seed)):
                with open('modelSave' + seed[s] + '/model_' + modelStruct[m] + '_' + alpha[a] + '_' + loss[l] + '/Analyzer.p', 'rb') as pickle_file:
                    listAnalyzer[s] = pickle.load(pickle_file)
            Analyzer = ACEAnalyzer.merge_ACEAnalyzers(listAnalyzer)
            pickle.dump(Analyzer, open('SumAnalyzer/' + modelStruct[m] + '_' + alpha[a] + '_' + loss[l] + '_' + 'Analyzer.p', "wb"))
            print('\nAnalyze on model ' +  modelStruct[m] + '_' + alpha[a] + '_' + loss[l])
            StatsErrorsSubstitutions = Analyzer.stats_errors_substitutions(stats_on_errors_only = True)
            print("\nSTATS ERROR SUBSTITUTIONS:\n------")
            print("Errors explained by substitutions rules: {}% of total errors\n------".format(round(Analyzer.total_errors_explained_by_substitutions*100.0/Analyzer.total_errors,2)))
            print("DETAIL ERRORS EXPLAINED BY SUBSTITUTION RULES:")
            for error_type, stat in StatsErrorsSubstitutions.items():
            	if stat*100 > 1:
            		print("{}: {}%".format(error_type, round(100*stat, 2)))
            
            
            # print(Analyzer.total_errors_degrees)
            # print(Analyzer.total_errors_when_non_diatonic_target)
            # print(Analyzer.total_non_diatonic_target)
            # print(Analyzer.degrees_analysis)
            StatsErrorsDegrees = Analyzer.stats_errors_degrees(stats_on_errors_only = True)
            print("\nSTATS ERROR DEGREES:\n------")
            print("Errors when the target is not diatonic: {}% ".format(round(Analyzer.total_errors_when_non_diatonic_target*100.0/Analyzer.total_non_diatonic_target,2)))
            print("Non diatonic target in {}% of the total errors".format(round(Analyzer.total_errors_when_non_diatonic_target*100.0/Analyzer.total_errors,2)))
            print("When relevant: incorrect degrees (modulo inclusions): {}% of total errors\n------".format(round(Analyzer.total_errors_degrees*100.0/Analyzer.total_errors,2)))
            print("DETAIL ERRORS OF DEGREES (modulo inclusions) WHEN THE TARGET IS DIATONIC:")
            for error_type, stat  in StatsErrorsDegrees.items():
            	if stat*100 > 1:
            		print("{}: {}%".format(error_type, round(100*stat,2)))
                        
#%%
with open('modelSave/model_conv3article_a5_tonnetz/Analyzer.p', 'rb') as pickle_file:
    Analyzer = pickle.load(pickle_file)
print(Analyzer)
#%%
from matplotlib import pyplot as plt
for i in range(3):
    plt.figure()
    plt.boxplot(res[i:9:3].T)
#%%
import librosa.display
import jams
item = '12_-_Polythene_Pam'
features= 'working/chords/pump/'
refs= 'dataset/isophonics/metadataTest/Beatles'


#jams = jams.load('{}/{}.jams'.format(refs, item), validate=False)
datum = np.load('{}/{}.npz'.format(features, item))['cqt/mag']

ann = jams.Annotation('chord')
ann_true = jams.Annotation('chord')
confidence = 1
print(item)
#load and transform jam
fname = json.load(open('dataset/isophonics/metadataTest/Beatles/'+ item + ".jams"))


u = fname
for nbacc in range(len(u['annotations'][0]['data'])):
    t_start = u['annotations'][0]['data'][nbacc]["time"]
    t_end = u['annotations'][0]['data'][nbacc]["time"]+u['annotations'][0]['data'][nbacc]["duration"]
    #vd = reduChord(reduChord(u['annotations'][0]['data'][nbacc]["value"], alpha2),'aMirex') #reduire la true annotation est à tester
    #vd = reduChord(u['annotations'][0]['data'][nbacc]["value"], 'reduceWOmodif') #reduire la true annotation est à tester
    vd = reduChord(u['annotations'][0]['data'][nbacc]["value"], 'reduceWOmodif')
    ann_true.append(time=t_start,
                       duration=t_end-t_start,
                       value=vd,
                       confidence=float(confidence))

maxFrame = len(datum[0])
for numFrame in range(maxFrame - transformOptions["contextWindows"]  + 1):
    #select frame in sample
    start = numFrame + (transformOptions["contextWindows"]  + 1)/2
    #get in time
    t_start, t_end = librosa.core.frames_to_time([start, start+1],
                                        sr=sr,
                                        hop_length=hop_length)
    #get predicted value
    #vd = reduChord(listChord[np.argmax(model.predict(datum[:,numFrame:numFrame+15]))],'aMirex')
    vd = listChord[np.argmax(model.predict(datum[:,numFrame:numFrame+15]))]
    ann.append(time=t_start,
                       duration=t_end-t_start,
                       value=vd,
                       confidence=float(confidence))

#%%
import jams.display
plt.figure(figsize=(10, 8))

ax = plt.subplot(2,1,1)
librosa.display.specshow(datum[0, :, :, 0].T,
                         sr=sr,
                         hop_length=hop_length,
                         x_axis='time')

plt.subplot(2,1,2, sharex=ax)
jams.display.display(ann_true, meta=False, label='Reference', alpha=0.5)
jams.display.display(ann, meta=False, label='Estimate', alpha=0.5)
plt.legend()
plt.tight_layout()