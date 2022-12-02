import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from time import sleep
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import concatenate, Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, Embedding, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, Sequence
from imblearn.under_sampling import *
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences


es = EarlyStopping(patience=3)
cl = CSVLogger('./logs/train.log')
tb = TensorBoard('./logs', write_images=True)
mc = ModelCheckpoint(filepath='./checkpoint',save_weights_only=True, save_best_only=True)


vocab_size = 30000 # 사용할 단어의 개수 (빈도 순)
N = 12 # 분류할 label의 종류
usage = 0.001 # 현재 data의 크기가 커서, generator를 만들기 전까지 이걸로 사용
max_len = 520 # padding을 맞출 문장의 최대 길이
embedding_dim = 100 # 임베딩 차원
dropout_rate = 0.3
batch_size = 4
tlf_dict = {} 
epoch = 32


while True:
    try:
        tlf_dict = pd.read_pickle("./data/tlf_dict_2")
        break
    except:
        print("Waiting for tlf_dict_2")
        sleep(600)
        pass


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

def data_generator(xs, x_txts, ys):
    for x, x_txt, y in zip(xs, x_txts, ys):
        yield ([x, x_txt], y)

class DataGenerator(Sequence):
    def __init__(self, X, y, vocab_size, tokenizer, batch_size, max_len, shuffle = True):
        self.X = X
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.length = len(X)
        self.shuffle = shuffle
        self.max_len = max_len
        self.y = y
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.floor(self.length / self.batch_size))
    
    def __data_generation(self, i):

        img = texts_to_tlf_once(self.X[i])
        txt = self.X[i]
        label = self.y[i]        
        return img, txt, label

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        img_data = []
        txt_data = []
        label_data = []

        for i in indexes:
            img, txt, label = self.__data_generation(i)
            img_data.append(img)
            txt_data.append(txt)
            label_data.append(label)
        txt_data = self.tokenizer.texts_to_sequences(txt_data)
        X_txt = pad_sequences(txt_data, maxlen=self.max_len)
        X_img = np.array(img_data)
        X_img = X_img.reshape((*X_img.shape, 1))
        Y_img = np.array(label_data)

        return [X_img, X_txt], Y_img


xs = x_train_undersamp
ys = y_train_undersamp

# shuffle False, random_state = 77 로 x_unuse와 x_use 고정
#x_unuse, x_use, y_unuse, y_use = train_test_split(x_train_undersamp, y_train_undersamp, test_size=usage, random_state=77, shuffle=False)
x_train, x_valid, y_train, y_valid = train_test_split(xs, ys, test_size=0.05, random_state=77, shuffle=True)
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)
Y_train = np.array(make_categorical(y_train))
Y_valid = np.array(make_categorical(y_valid))

train_generator = DataGenerator(x_train, Y_train, vocab_size=vocab_size, tokenizer=tokenizer, 
                                batch_size=batch_size, max_len=max_len, shuffle=True)
valid_generator = DataGenerator(x_valid, Y_valid, vocab_size=vocab_size, tokenizer=tokenizer, 
                                batch_size=batch_size, max_len=max_len, shuffle=True)

'''
X_train = np.array(texts_to_tlf(x_train))
X_valid = np.array(texts_to_tlf(x_valid))
X_train = X_train.reshape((*X_train.shape,1))
X_valid = X_valid.reshape((*X_valid.shape,1))
x_train_txt = tokenizer.texts_to_sequences(x_train)
x_valid_txt = tokenizer.texts_to_sequences(x_valid)
X_train_txt = pad_sequences(x_train_txt, maxlen=max_len)
X_valid_txt = pad_sequences(x_valid_txt, maxlen=max_len)
'''

'''
next = train_generator.__getitem__(3)
print(type(next[0][0]), type(next[0][1]), type(next[1]))
try:
    print(next[0][0].shape)
    print(next[0][1].shape)
    print(next[1].shape)
except:
    pass

while True:pass
'''

txtInput = Input(shape=(max_len,), name='txt_input')
txt = Embedding(vocab_size, embedding_dim, input_length=max_len)(txtInput)
txt = GRU(512, return_sequences=True)(txt)
txt = Dropout(dropout_rate)(txt)
txt = GRU(512, return_sequences=True)(txt)
txt = Dropout(dropout_rate)(txt)
txt = GRU(512)(txt)
txt = Dense(512, activation='relu')(txt)

imgInput = Input(shape=(vocab_size, N, 1, ), name = 'img_input')
img = Conv2D(4, (3,3), padding='same', name='block1_conv1', activation='relu')(imgInput)
img = Conv2D(4, (3,3), padding='same', name='block1_conv2', activation='relu')(img)
#img = Conv2D(64, (3,3), padding='same', name='block1_conv3', activation='relu')(img)
img = MaxPooling2D((4,1), name='block1_pool')(img)
img = Conv2D(8, (3,3), padding='same', name='block2_conv1', activation='relu')(img)
img = Conv2D(8, (3,3), padding='same', name='block2_conv2', activation='relu')(img)
#img = Conv2D(128, (3,3), padding='same', name='block2_conv3', activation='relu')(img)
img = MaxPooling2D((4,1), name='block2_pool')(img)
img = Conv2D(16, (3,3), padding='same', name='block3_conv1', activation='relu')(img)
img = Conv2D(16, (3,3), padding='same', name='block3_conv2', activation='relu')(img)
#img = Conv2D(256, (3,3), padding='same', name='block3_conv3', activation='relu')(img)
img = MaxPooling2D((4,1), name='block3_pool')(img)
img = Conv2D(33, (3,3), padding='same', name='block4_conv1', activation='relu')(img)
img = Conv2D(32, (3,3), padding='same', name='block4_conv2', activation='relu')(img)
img = MaxPooling2D((4,1), name='block4_pool')(img)
img = Conv2D(64, (3,3), padding='same', name='block5_conv1', activation='relu')(img)
img = Conv2D(64, (3,3), padding='same', name='block5_conv2', activation='relu')(img)
img = MaxPooling2D((4,1), name='block5_pool')(img)
img = Flatten()(img)

output = concatenate([img, txt], axis=-1)
output = Dropout(dropout_rate)(output)
output = Dense(N, activation='softmax')(output)

model = Model(inputs = [imgInput, txtInput], outputs = output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

model.summary()

#history = model.fit([X_train, X_train_txt], Y_train, validation_data=([X_valid, X_valid_txt], Y_valid), workers=-1, epochs = 1000, callbacks=[es, cl, tb, mc])
history = model.fit(x = train_generator, validation_data=valid_generator, workers=-1, epochs = epoch, callbacks=[es, cl, tb, mc])
with open('./trainHistory', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
