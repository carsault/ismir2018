# Using musical relationships between chord labels in Automatic Chord Extraction

This repository contains the source code, results, and paper for the ISMIR 2018 submission, "Using musical relationships between chord labels in Automatic Chord Extraction" (Carsault, Nika & Esling).

The audio dataset must be put into the dataset/audio file.

Install python 3.X
Install dependant libraires

To train a new model:
	python3 chordTrain.py -o [option1,option2,option3,option4]

	option1: Kind of model (default: conv3article)
		other models could be used or implemented in the file utilities/models.py

	option2: Alphabet (default: a0)
		a0: Major/Minor alphabet
		a2: correpsonds to the alphabet a1 in the article
		a5: corresponds to the alphabet a2 in the article

	option3: distance for learning (default: categorical_crossentropy)
		categorical_crossentropy: binary classification
		tonnetz: distance based on tonnetz distance
		euclidian: distance based on euclidian distance

	option4: seed for random split of the training/testing dataset

	After the training the model will be saved in the folder 'modelSave[option4]/model_[option0]_[option1]_[option3]
		in this same folder we will find other files informing on the split and the learning loss evolution through the training

To test existing models:
	If you want to test for one model, uncomment and run the "Stat for one model" part in the code testChordModel.py
	If you want to test over several model configurations and with different seeds, run testChordModel.py part by part in order to print all the different plots


