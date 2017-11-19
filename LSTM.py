
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(7)

# Our dataset is preprocessed such that every word is replaced by a unique integer representing its popularity (frequency in the document). The Lesser the
# number the more the frequency. For eg word which has most frequency is assigned number 0. Thus our sentences are a sequence of Integers.

# Vocablary Size
top_words = 500
# This denotes the frequency wise top 5000 numbers (0 - 4999)

# We will divide the dataset into two parts 25000 each examples already stored in imdb dataset
(X_train, Y_train),(X_test, Y_test) = imdb.load_data(num_words=top_words)
# Now are vocabalry contains only 5000 distinct words. Where as total number of words maybe more or equal.

# Length of reviews are different lets make it uniform. Each review is our sequence and number of words represent TIMESTEPS in LSTM. Thats why we make it same
max_len_of_review = 500

# Thus we use Keras.Preprocessing.sequence to pad the reviews
X_train = sequence.pad_sequences(X_train, maxlen = max_len_of_review)
X_test = sequence.pad_sequences(X_test, maxlen = max_len_of_review)

# Creating Model

# Encoded vector lenght
enc_len = 32

model = Sequential()
model.add(Embedding(input_dim=top_words, output_dim=enc_len, input_length=max_len_of_review))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))

# Compiling Model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])     # Loss fucntion (logarithmic loss func) and optimizer adam , may use SGD

model.fit(X_train,Y_train,epochs=2, batch_size=64)

# Evaluation
score = model.evaluate(X_test,Y_test,verbose=1)

print ("Accuracy: %.2f%%" % (score[1]*100))