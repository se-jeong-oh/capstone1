import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


vocab_size = 10000
num_classes = 12
max_len = 500
check_label = 10
max = 30000
cur = 0
tlf_dict = {}
N = 12

def texts_to_tlf_once(x):
    # input : tokenized 된 문장 (noun으로 이루어진)
    # output max_len * N 사이즈의 2차원 배열
    tlf2d_vector = np.zeros((max_len, N))
    curr = 0
    for word in x:
        try:
            for i in range(N):
                tlf2d_vector[curr][i] += tlf_dict[word][i]
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

train_load_df = pd.read_pickle("./data/train_df.pkl")
tlf_dict = pd.read_pickle("./data/tlf_dict")
train_sent = train_load_df['sentence']
train_label = train_load_df['label']

x_unuse, x_use, y_unuse, y_use = train_test_split(train_sent, train_label, test_size=0.2, random_state=1, shuffle=True)
x_train, x_valid, y_train, y_valid = train_test_split(x_use, y_use, test_size=0.1, random_state=2, shuffle=True)
#x_train, x_valid, y_train, y_valid = train_test_split(train_sent, train_label, test_size=0.1, random_state=1, shuffle=True)

'''
ttokenizer = Tokenizer(num_words = vocab_size)
ttokenizer.fit_on_texts(x_train)
X_train = ttokenizer.texts_to_sequences(x_train)

word_dic = ttokenizer.index_word

lab = [0 for _ in range(N)]
for y in y_train:
    lab[y] += 1
# 특정 label에 많고, 다른 label에 적다면 LF 값 높음
# 특정 label에 적고, 다른 label에 많다면 이 역시 LF 값 높음
# 모든 label에서 적거나, 모든 label에서 많으면 LF 값이 낮음
for idx, word in word_dic.items():
    # idx : 1, 2, ... etc
    # word : 표지, 비상, ... 등
    ilf = [0 for _ in range(N)]
    freq_lab = [0 for _ in range(N)]
    port = [0 for _ in range(N)]
    #freq_lab = 0
    #freq_else = 0
    for x, y in zip(x_train, y_train):
        for x_word in x:
            if x_word == word:
                freq_lab[y] += 1
                break
    for i in range(N) :
        # 특정 단어가 i 라벨에 있는 문서들에 포함된 비율
        port[i] = freq_lab[i] / lab[i]
    for i in range(N):
        avg_freq = 0
        for j in range(N):
            if i == j: continue
            avg_freq += port[j]
        avg_freq /= (N-1)
        ilf[i] = abs(avg_freq - port[i])
    ilf_dict[word] = ilf
    cur += 1
    print("Preprocessing : ", cur, "/", 29358)
    if cur == max:
        break
try:
    with open('./data/tlf_dict', 'wb') as fp:
        pickle.dump(ilf_dict, fp)
except:
    pass
'''
X_train = np.array(texts_to_tlf(x_train))
X_valid = np.array(texts_to_tlf(x_valid))
Y_train = np.array(make_categorical(y_train))
Y_valid = np.array(make_categorical(y_valid))

X_train = X_train.reshape((*X_train.shape,1))
X_valid = X_valid.reshape((*X_valid.shape,1))

txtInput = Input(shape=(max_len, N, 1, ), name = 'txt_input')
img = Conv2D(64, (3,3), padding='same', name='block1_conv1', activation='relu')(txtInput)
img = Conv2D(64, (3,3), padding='same', name='block1_conv2', activation='relu')(img)
img = Conv2D(64, (3,3), padding='same', name='block1_conv3', activation='relu')(img)
img = MaxPooling2D((2,2), name='block1_pool')(img)
img = Conv2D(128, (3,3), padding='same', name='block2_conv1', activation='relu')(img)
img = Conv2D(128, (3,3), padding='same', name='block2_conv2', activation='relu')(img)
img = Conv2D(128, (3,3), padding='same', name='block2_conv3', activation='relu')(img)
img = MaxPooling2D((2,2), name='block2_pool')(img)
#img = Conv2D(512, (3,3), padding='same', name='block4_conv1', activation='relu')(img)
#img = Conv2D(512, (3,3), padding='same', name='block4_conv2', activation='relu')(img)
#img = MaxPooling2D((4,4), name='block4_pool')(img)
#img = Conv2D(1024, (3,3), padding='same', name='block5_conv1', activation='relu')(img)
#img = Conv2D(1024, (3,3), padding='same', name='block5_conv2', activation='relu')(img)
#img = MaxPooling2D((4,4), name='block5_pool')(img)
img = Flatten()(img)
output = Dense(N, activation='softmax')(img)

model = Model(inputs = txtInput, outputs = output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()

history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), workers=-1, epochs = 50)

with open('./trainHistory', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)