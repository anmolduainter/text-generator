# Importing Libraries

import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
import argparse

# Setting up the variables

# DATA_DIR
DATA_DIR = './shakespeare_input.txt'

#BATCH SIZE
BATCH_SIZE = 5000

#Hidden Layer Dimension
HIDDEN_DIM = 500

# Sequence Length
SEQ_LENGTH = 100

#Weights
WEIGHTS = 'checkpoint_layer_2_hidden_500_epoch_45.hdf5'

#Generate Length
GENERATE_LENGTH = 500

#Number of hidden layers
LAYER_NUM = 2

# Mode 
MODE = 'train'

print(LAYER_NUM)

# function for generating text 
def generate_text(model, length, vocab_size, ix_to_char):

	# Taking Random number from vocabulary size
	ix = [np.random.randint(vocab_size)]
     
     # getting char value of random number 
	y_char = [ix_to_char[ix[-1]]]
 
     # Making of Numpy Array having shape given below
     # Here length is GENERATE_LENGTH
	X = np.zeros((1, length, vocab_size))
 
     # Taking i till we want to generate that is GENERATE LENGTH
	for i in range(length):
		# appending the last predicted character to sequence
		X[0, i, :][ix[-1]] = 1
		print(ix_to_char[ix[-1]], end="")
		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
		y_char.append(ix_to_char[ix[-1]])
	return ('').join(y_char)



# method for preparing the training data
def load_data(data_dir, seq_length):
    
    # Opening The respective file from directory
	data = open(data_dir, 'r').read()

    # getting a list of unique characters
	chars = list(set(data))

    # getting the length of list of unique characters
	VOCAB_SIZE = len(chars)

    # Printing the number of characters inside file
	print('Data length: {} characters'.format(len(data)))

    # Printing the number of unique characters inside the file
	print('Vocabulary size: {} characters'.format(VOCAB_SIZE))
 
    # Building up a dictionary where keys are no. and values are unique chars
	ix_to_char = {ix:char for ix, char in enumerate(chars)}

    # Building up a dictionary where keys are unique chars and values are numbers
	char_to_ix = {char:ix for ix, char in enumerate(chars)}

    # We’re gonna use Keras to create and train our Network, 
    # so we must convert the data into this form:
    #  (number_of_sequences, length_of_sequence, number_of_features).

    # VOCAB_SIZE = number of features
    
	X = np.zeros((len(data)//seq_length, seq_length, VOCAB_SIZE))
	y = np.zeros((len(data)//seq_length, seq_length, VOCAB_SIZE))

	for i in range(0, len(data)//seq_length):
         # steps for filling X
        # Taking chars whose length is equal to sequence length from the file store in data 
		X_sequence = data[i*seq_length:(i+1)*seq_length]

        # Taking all the chars from X_sequence and converting them to respective numbers 
        # by seeing through dictionary char -> no.
		X_sequence_ix = [char_to_ix[value] for value in X_sequence]

        # this is a 2d array with row length = sequence length and col length = Vocabulary size
		input_sequence = np.zeros((seq_length, VOCAB_SIZE))

		for j in range(seq_length):
              
              # Setting the value of 2darray where row is j and col is value at X_sequence_ix[j] to be 1
			input_sequence[j][X_sequence_ix[j]] = 1
              # storing the value of input_squence at X[i]
			X[i] = input_sequence

        # Same steps for y also 
		y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
		y_sequence_ix = [char_to_ix[value] for value in y_sequence]
		target_sequence = np.zeros((seq_length, VOCAB_SIZE))
		for j in range(seq_length):
			target_sequence[j][y_sequence_ix[j]] = 1.
			y[i] = target_sequence
   
      #Returning 2 arrays X and y and Vocabulary size and one dict = no->char
	return X, y, VOCAB_SIZE, ix_to_char


# Creating training data
X, y, VOCAB_SIZE, ix_to_char = load_data(DATA_DIR, SEQ_LENGTH)

# Creating and compiling the Network (LSTM)
model = Sequential()
model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))

#Adding More Hidden Layers
for i in range(LAYER_NUM - 1):
  model.add(LSTM(HIDDEN_DIM, return_sequences=True))
model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# Generate some sample before training to know how bad it is!
generate_text(model,500, VOCAB_SIZE, ix_to_char)

# if there is already pretrained weights
if not WEIGHTS == '':
  # Load Weights from pretrained model  
  model.load_weights(WEIGHTS)
  # getting the number of epochs that has been already done from the name of the file
  nb_epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
#if not a pretrained model
else:
  #Number of epochs = 0  
  nb_epoch = 0

# Training if there is no trained weights specified and mode is train
if MODE == 'train' or WEIGHTS == '':

  #Making of a loop
  while True:
    # Printing the number of epochs that has been passed  
    print('\n\nEpoch: {}\n'.format(nb_epoch))
    
    # Fillting the LSTM model 
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, nb_epoch=50)
    
    # Incrementing the number of epochs by 1
    nb_epoch += 1
    
    #Checking if number of epochs is divisible by 5 then save the weights of trained model
    if nb_epoch % 5 == 0:
      model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, nb_epoch))

# Else, loading the trained weights and performing generation only
elif WEIGHTS == '':
  # Loading the trained weights
  model.load_weights('checkpoint_layer_2_hidden_500_epoch_55.hdf5')
  #generating the text with the pretrained model
  generate_text(model,GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
  print('\n\n')
else:
  print('\n\nNothing to do!')
