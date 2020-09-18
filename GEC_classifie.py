from __future__ import print_function, division
from builtins import range

import os
import sys,re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, GlobalMaxPool1D, Bidirectional
from tensorflow.keras.layers import LSTM, MaxPooling1D, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score


# some configuration
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
EPOCHS = 10



# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('/home/hassan/Documents/machine_learning_examples/nlp_class3/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))



# prepare text samples and their labels
print('Loading in Data...')

sentences = np.empty(0,dtype='object')
targets = np.zeros((0,28))
labels = {"Vt":0,"Vm":1,"V0":2,"Vform":3,"SVA":4,"ArtOrDet":5,"Nn":6,"Npos":7,"Pform":8,"Pref":9,"Prep":10,"Wci":11,"Wa":12,"Wform":13,"Wtone":14,"Srun":15,"Smod":16,"Spar":17,"Sfrag":18,"Ssub":19,"WOinc":20,"WOadv":21,"Trans":22,"Mec":23,"Rloc":24,"Cit":25,"Others":26,"Um":27}
i=0
f=open('/home/hassan/Documents/machine_learning_examples/nlp_class3/toxic/conll14st-preprocessed.m2')
line = f.readline()
while line:
    line = line.strip()
    if line:
        # print(line)
        if re.search("^S",line):
            a=np.empty(1,dtype='object')
            a[0]=line[2:]
            sentences=np.concatenate([sentences,a])
            b=np.zeros((1,28))
            targets=np.concatenate([targets,b])
            print(i)
            i=i+1
        elif re.search("^A",line):
            # print("found annotation")
            for key in labels:
                # print(labels.get(key))
                if re.search(key,line):
                    # print(labels.get(key))
                    targets[i-1,labels.get(key)]=1
                    break
    line=f.readline()
# print(sentences)
print(targets[3])
print(targets[57139])
print("found line", i)


# convert the sentences (strings) into integers
print('Tokenization...')
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
print("sequence[0]: ", sequences[0])

print("max sequence length:", max(len(s) for s in sequences))
print("min sequence length:", min(len(s) for s in sequences))
# s = sorted(len(s) for s in sequences)
# print("median sequence length:", s[len(s) // 2])
#
# print("max word index:", max(max(seq) for seq in sequences if len(seq) > 0))
#
#
# get word -> integer mapping
word2idx = tokenizer.word_index
print('Found %s unique tokens.' % len(word2idx))
# print(next(iter(word2idx)))
# exit()

# pad sequences so that we get a N x T matrix
print('Pad sequences...')
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("data[0]", data[0])
print('Shape of data tensor:', data.shape)



# prepare embedding matrix
print('Filling pre-trained embeddings...')
print("len word2idx", len(word2idx))
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector

print("embedding_matrix shape:", embedding_matrix.shape)
print(word2idx.get("creating"))
print("embedding_matrix[1113]", embedding_matrix[1113])



# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
print('Building Embedding layer ...')
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False
)


print('Building model...')

# create an LSTM network with a single LSTM
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
# x = LSTM(15, return_sequences=True)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
output = Dense(28, activation="sigmoid")(x)

model = Model(input_, output)
model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(lr=0.01),
  metrics=['accuracy'],
)

# train a 1D convnet with global maxpooling
# input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
# x = embedding_layer(input_)
# x = Conv1D(128, 3, activation='relu')(x)
# x = MaxPooling1D(3)(x)
# x = Conv1D(128, 3, activation='relu')(x)
# x = MaxPooling1D(3)(x)
# x = Conv1D(128, 3, activation='relu')(x)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# output = Dense(28, activation='sigmoid')(x)

# model = Model(input_, output)
# model.compile(
#   loss='binary_crossentropy',
#   optimizer='rmsprop',
#   metrics=['accuracy']
# )

print('Training model...')
r = model.fit(
  data,
  targets,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=VALIDATION_SPLIT
)


# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# plot the mean AUC over each label
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))
