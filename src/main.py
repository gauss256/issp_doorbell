from __future__ import print_function
from __future__ import division

import os
import time
import pickle
import numpy as np
import sys

from PIL import Image
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json

from sklearn.metrics import log_loss, accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

from utilities import write_submission, calc_geom, calc_geom_arr, mkdirp

DOWNSAMPLE = 8
NUM_CLASSES = 2
RANDOM_STATE = 67
TESTING = False

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/data_{}.pkl'.format(DOWNSAMPLE) if not TESTING else 'dataset/data_{}_subset.pkl'.format(DOWNSAMPLE))

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', 'summaries/')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')

mkdirp(CHECKPOINT_PATH)
mkdirp(SUMMARY_PATH)
mkdirp(MODEL_PATH)

NB_EPOCHS = 10 if not TESTING else 1
MAX_FOLDS = 3 if not TESTING else 2

WIDTH, HEIGHT, NB_CHANNELS = 352 // DOWNSAMPLE, 240 // DOWNSAMPLE, 1
BATCH_SIZE = 16  # [50]

with open(DATASET_PATH, 'rb') as f:
    X_train_raw, y_train_raw, X_test, y_test, X_test_ids = pickle.load(f)    
    if NB_CHANNELS == 1:
        X_train_raw = X_train_raw.reshape(X_train_raw.shape[0], 1, WIDTH, HEIGHT)
        X_test = X_test.reshape(X_test.shape[0], 1, WIDTH, HEIGHT) 

predictions_total = [] # accumulated predictions from each fold
scores_total = [] # accumulated scores from each fold
num_folds = 0

def filereader(fname):
	x = np.array(Image.open(fname))
	
	if len(x.shape) == 2:
		#add an additional colour dimension if the only dimensions are width and height
		return preprocess( x.reshape((1, 1) + x.shape) )
	if len(x.shape) == 3:
		return preprocess( x.reshape((1, 0) + x.shape) )	
	
def preprocess(X):
	#this preprocessor crops one pixel along each of the sides of the images
	#return X[:, :, 1:-1, 1:-1] / 255.0	
	return X/ 255.0	

def myGenerator(y, chunk_size, batch_size, fnames):

	#read and preprocess first file to figure out the image dimensions
	
	#sample_file = filereader('train/train' + str(0) + '.png')
	#sample_file = np.array(Image.open('sample.jpg'))
        sample_file = filereader('sample.jpg')
	new_img_colours, new_img_rows, new_img_cols = sample_file.shape[1:]
	
	pool = Pool(processes=16)
	
	while 1:
		for i in xrange(np.ceil(1.0 * len(fnames) / chunk_size).astype(int)):
			this_chunk_size = len(fnames[i * chunk_size : (i+1) * chunk_size])
			X = pool.map(filereader, [fnames[i * chunk_size + i2]  for i2 in xrange(this_chunk_size)])
			X = np.array(X).reshape((-1, new_img_colours, new_img_rows, new_img_cols)) #.astype('float32')
			#if 'test' in fnames[0]:
			print(str(i))			
						
			for j in xrange(int(np.floor(this_chunk_size/batch_size))): 
				#training set
				if not y is None:	
					yield X[j*batch_size:(j+1)*batch_size], y[i*chunk_size+j*batch_size:i*chunk_size+(j+1)*batch_size]
				#test set
				else:
					yield X[j*batch_size:(j+1)*batch_size] 

def vgg_bn():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', input_shape=(NB_CHANNELS, WIDTH, HEIGHT)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax', init='he_normal'))
    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_mnist():
    nb_filters = 32
    nb_conv = 3
    nb_pool = 2
    nb_classes = 2
    
    model = Sequential()

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(NB_CHANNELS, WIDTH, HEIGHT)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
    return model

for train_index, valid_index in KFold(len(X_train_raw), n_folds=MAX_FOLDS, shuffle=True, random_state=RANDOM_STATE):
    print('Fold {}/{}'.format(num_folds + 1, MAX_FOLDS))

    X_train, y_train = X_train_raw[train_index,...], y_train_raw[train_index,...]
    X_valid, y_valid = X_train_raw[valid_index,...], y_train_raw[valid_index,...]

    model = vgg_bn()

    model_path = os.path.join(MODEL_PATH, 'model_{}.json'.format(num_folds))
    with open(model_path, 'w') as f:
        print('Writing model...')
        f.write(model.to_json())

    checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}.h5'.format(num_folds))
    #!!! why?
    if 0:
        # restore existing checkpoint, if it exists
        if os.path.exists(checkpoint_path):
            print('Restoring fold from checkpoint.')
            model.load_weights(checkpoint_path)
    #!!!

    summary_path = os.path.join(SUMMARY_PATH, 'model_{}'.format(num_folds))
    mkdirp(summary_path)

    callbacks = [
        #EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto'),
        #ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        #TensorBoard(log_dir=summary_path, histogram_freq=0)
    ]
    print('Starting model fit...')
    generator = False
    if generator:
        chunk_size = 15000
        # assumption : fnames_train is a list of all training images
        fnames_train = os.listdir('/home/issp/Doorbell/images/train/c0')
        fnames_train = fnames_train + os.listdir('/home/issp/Doorbell/images/train/c1')
        print (fnames_train)
        model.fit_generator(
            myGenerator(y_train, chunk_size, BATCH_SIZE, fnames_train), 
            samples_per_epoch = y_train.shape[0], 
            nb_epoch = NB_EPOCHS, 
            verbose = 2, 
            callbacks = [], 
            validation_data = None, 
            class_weight = None 
        ) # max_q_size = 10
    else:
        model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE, nb_epoch=NB_EPOCHS,
            shuffle=True,
            verbose=1,
            validation_data=(X_valid, y_valid),
            callbacks=callbacks
        )

    print('Finished model fit...')

    predictions_valid = model.predict(X_valid, batch_size=100, verbose=1)
    score_valid = log_loss(y_valid, predictions_valid)
    scores_total.append(score_valid)

    print('Score: {}'.format(score_valid))

    predictions_test = model.predict(X_test, batch_size=100, verbose=1)
    predictions_total.append(predictions_test)

    num_folds += 1

score_geom = calc_geom(scores_total, MAX_FOLDS)
predictions_geom = calc_geom_arr(predictions_total, MAX_FOLDS)
score_test = log_loss(y_test, predictions_geom)
y_predict = np.array(predictions_geom)
y_predict = np.argmax(y_predict, axis=1)
acc_test = accuracy_score(y_test, y_predict)

print()
print('Overall score: {}'.format(score_test))
print('Overall accuracy: {:.1f}%'.format(acc_test * 100.0))

submission_path = os.path.join(SUMMARY_PATH, 'submission_{}_{:.2}.csv'.format(int(time.time()), score_geom))
write_submission(predictions_geom, X_test_ids, submission_path)
