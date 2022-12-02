import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from imblearn.under_sampling import *
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, ModelCheckpoint


es = EarlyStopping(mid_delta=0, patience=10)
cl = CSVLogger('./logs/train.log')
tb = TensorBoard('./logs', write_images=True)
mc = ModelCheckpoint(filepath='./checkpoint',save_weights_only=True, save_best_only=True)


vocab_size = 50000 # 사용할 단어의 개수 (빈도 순)
N = 12 # 분류할 label의 종류
usage = 0.8 # 현재 data의 크기가 커서, generator를 만들기 전까지 이걸로 사용
max_len = 500

tlf_dict = pd.read_pickle("./data/tlf_dict_2")
tlf_tuplelist = list(tlf_dict.items())
tlf_tuplelist_keys = [key[0] for key in tlf_tuplelist]

x_train_undersamp = pd.read_pickle("./data/x_train_undersamp")
y_train_undersamp = pd.read_pickle("./data/y_train_undersamp")

def texts_to_tlf_once(x):
    # input : tokenized 된 문장 (noun으로 이루어진)
    # output vocab_size * N 사이즈의 2차원 배열
    tlf2d_vector = np.zeros((vocab_size, N))    
    for word in x:
        try:
            idx = tlf_tuplelist_keys.index(word)
            for i in range(N):
                try:                
                    tlf2d_vector[idx][i] += tlf_dict[word][i]
                except:
                    break
        except:
            continue
    return tlf2d_vector

def texts_to_tlf(x_train):
    X_train = []
    for x in x_train:
        tlf2d_vector = texts_to_tlf_once(x)
        X_train.append(tlf2d_vector)
    return X_train

def make_categorical(y_train):
    Y_train = []
    for y in y_train:
        y_init = [0 for _ in range(N)]
        y_init[y] = 1
        Y_train.append(y_init)
    return Y_train



# shuffle False, random_state = 77 로 x_unuse와 x_use 고정
x_unuse, x_use, y_unuse, y_use = train_test_split(x_train_undersamp, y_train_undersamp, test_size=usage, random_state=77, shuffle=False)
x_train, x_valid, y_train, y_valid = train_test_split(x_use, y_use, test_size=0.1, random_state=77, shuffle=True)


X_train = np.array(texts_to_tlf(x_train))
X_valid = np.array(texts_to_tlf(x_valid))
Y_train = np.array(make_categorical(y_train))
Y_valid = np.array(make_categorical(y_valid))

X_train = X_train.reshape((*X_train.shape,1))
X_valid = X_valid.reshape((*X_valid.shape,1))

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)

X_train_txt = tokenizer.texts_to_matrix(x_train)

txtInput = Input(shape=())
imgInput = Input(shape=(vocab_size, N, 1, ), name = 'img_input')
img = Conv2D(64, (3,3), padding='same', name='block1_conv1', activation='relu')(imgInput)
img = Conv2D(64, (3,3), padding='same', name='block1_conv2', activation='relu')(img)
#img = Conv2D(64, (3,3), padding='same', name='block1_conv3', activation='relu')(img)
img = MaxPooling2D((4,1), name='block1_pool')(img)
img = Conv2D(128, (3,3), padding='same', name='block2_conv1', activation='relu')(img)
img = Conv2D(128, (3,3), padding='same', name='block2_conv2', activation='relu')(img)
#img = Conv2D(128, (3,3), padding='same', name='block2_conv3', activation='relu')(img)
img = MaxPooling2D((4,1), name='block2_pool')(img)
img = Conv2D(256, (3,3), padding='same', name='block3_conv1', activation='relu')(img)
img = Conv2D(256, (3,3), padding='same', name='block3_conv2', activation='relu')(img)
#img = Conv2D(256, (3,3), padding='same', name='block3_conv3', activation='relu')(img)
img = MaxPooling2D((4,1), name='block3_pool')(img)
img = Conv2D(512, (3,3), padding='same', name='block4_conv1', activation='relu')(img)
img = Conv2D(512, (3,3), padding='same', name='block4_conv2', activation='relu')(img)
img = MaxPooling2D((4,1), name='block4_pool')(img)
img = Conv2D(1024, (3,3), padding='same', name='block5_conv1', activation='relu')(img)
img = Conv2D(1024, (3,3), padding='same', name='block5_conv2', activation='relu')(img)
img = MaxPooling2D((4,1), name='block5_pool')(img)
img = Flatten()(img)
output = Dense(N, activation='softmax')(img)

model = Model(inputs = imgInput, outputs = output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()

history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), workers=-1, epochs = 1000, callbacks=[es, cl, tb, mc])

with open('./trainHistory', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)