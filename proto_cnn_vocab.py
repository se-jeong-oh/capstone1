import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from time import sleep
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import concatenate, Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, Embedding, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, Sequence
from imblearn.under_sampling import *
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from keras.optimizers import SGD 
from scipy import sparse
from imblearn.over_sampling import SMOTE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

m_n = 'tfidf_cut5000_vocabsize30000_adam_len50'
model_name = './img/' + m_n + '.png'
file_path = './checkpoint/' + m_n + '_{epoch:03d}.ckpt'
log_path = './logs/' + m_n + '.log'
save_path = './h5/' + m_n + '.h5'
lambd = 0.001
vocab_size = 30000 # 사용할 단어의 개수 (빈도 순)
N = 12 # 분류할 label의 종류
usage = 0.001 # 현재 data의 크기가 커서, generator를 만들기 전까지 이걸로 사용
max_len = 520 # padding을 맞출 문장의 최대 길이
embedding_dim = 5 # 임베딩 차원
dropout_rate = 0.4
batch_size = 64
tlf_dict = {} 
epoch = 1000
pool = 8
data_amount = 5000
append_len = 150
conv = 1
cut_len = 25
mode = 'img'

cos_decay_ann = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=0.1, first_decay_steps=10, t_mul=2, m_mul=0.8, alpha=0)

es = EarlyStopping(patience=5)
cl = CSVLogger(log_path)
tb = TensorBoard('./logs', write_images=True)
mc = ModelCheckpoint(filepath=file_path,save_weights_only=True, save_best_only=True)
#lr = LearningRateScheduler(step_decay, verbose=1)
while True:
    try:
        tlf_dict = pd.read_pickle("./data/tlf_dict_3_30000")
        break
    except:
        print("Waiting for tlf_dict_2")
        sleep(600)
        pass


tlf_tuplelist = list(tlf_dict.items())
tlf_tuplelist_keys = [key[0] for key in tlf_tuplelist]

xs = pd.read_pickle('./data/x_train_oversamp')
ys = pd.read_pickle('./data/y_train_oversamp')

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

def texts_to_tlf_once_flat(x):
    # input : tokenized 된 문장 (noun으로 이루어진)
    # output vocab_size * N 사이즈의 1차원 배열
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
    tlf2d_vector = tlf2d_vector.reshape(1, -1)
    return list(tlf2d_vector)

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

def make_sentence(x):
    sentence = ''
    for word in x:
        sentence += word
        sentence += ' '
    return sentence

class DataGenerator(Sequence):
    def __init__(self, X, y, vocab_size, tokenizer, batch_size, max_len, shuffle = True, mode='imgtxt', smote=False):
        self.X = X
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.length = len(X)
        self.shuffle = shuffle
        self.max_len = max_len
        self.y = y
        self.mode = mode
        self.smote = smote
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.floor(self.length / self.batch_size))
    
    def __data_generation(self, i, mode='imgtxt', smote=False):
        if mode == 'imgtxt':
            img = texts_to_tlf_once(self.X[i])
            txt = self.X[i]
        elif mode == 'img':
            if not smote:
                img = texts_to_tlf_once(self.X[i])
            else:
                img = self.X[i]
            txt = 0
        elif mode == 'txt':
            img = 0
            txt = self.X[i]
        elif mode == 'tfidf':
            img = 0
            txt = self.X[i]
        elif mode == 'tfidfimg':
            img = texts_to_tlf_once(self.X[i])
            txt = self.X[i]
        elif mode == 'smote':
            img = self.X[i]
            txt = 0
            
        label = self.y[i]        
        return img, txt, label

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        img_data = []
        txt_data = []
        label_data = []

        for i in indexes:
            img, txt, label = self.__data_generation(i, self.mode, self.smote)
            img_data.append(img)
            txt_data.append(txt)
            label_data.append(label)
        Y_img = np.array(label_data)
        
        if self.mode == 'imgtxt':
            X_img = np.array(img_data)
            X_img = X_img.reshape((*X_img.shape, 1))
            txt_data = self.tokenizer.texts_to_sequences(txt_data)
            X_txt = pad_sequences(txt_data, maxlen=self.max_len)
            return [X_img, X_txt], Y_img
        
        elif self.mode == 'img':
            X_img = np.array(img_data)
            #X_img = X_img.reshape((*X_img.shape, 1))
            if not self.smote:
                X_img = X_img.reshape((-1, self.vocab_size * N))      
            return X_img, Y_img
        
        elif self.mode == 'txt':
            txt_data = self.tokenizer.texts_to_sequences(txt_data)
            #txt_data = self.tokenizer.texts_to_matrix(txt_data, mode='tfidf')
            X_txt = pad_sequences(txt_data, maxlen=self.max_len)
            return X_txt, Y_img
        
        elif self.mode == 'tfidf':            
            #txt_data = self.tokenizer.texts_to_matrix(txt_data, mode='tfidf')
            #print(len(txt_data[0]))
            #X_txt = pad_sequences(txt_data, maxlen=self.max_len)
            #print(len(X_txt))
            #debug_loop()
            X_txt = pad_sequences(txt_data, maxlen=vocab_size)
            return X_txt, Y_img
        
        elif self.mode == 'tfidfimg':
            txt_data = self.tokenizer.texts_to_matrix(txt_data, mode='tfidf')
            X_txt = pad_sequences(txt_data, maxlen=vocab_size)
            X_img = np.array(img_data)
            X_img = X_img.reshape((*X_img.shape, 1)) 
            
            return [X_img, X_txt], Y_img
        
        elif self.mode == 'smote':
            X_img = np.array(img_data)
            X_img = X_img.reshape((*X_img.shape, 1))
            return X_img, Y_img  
            
def debug_loop():
    while True: pass

def append_xy(xs, ys, length=append_len):
    app_x = []
    app_y = []
    
    queue = [[] for _ in range(N)]
    
    for x, y in zip(xs, ys):
        queue[y] += x
        if len(queue[y]) > length:
            app_x.append(queue[y])
            app_y.append(y)
            queue[y] = []
    for i in range(N):
        if len(queue[i]) != 0:
            app_x.append(queue[i])
            app_y.append(i)
            queue[y] = []
            
    return app_x, app_y
        
def augment_xy(x_train, y_train, minimum=None):
    lab = [0 for _ in range(N)]

    for y in y_train:
        lab[y] += 1

    x_train_list = list(x_train)
    y_train_list = list(y_train)

    if minimum == None:
        minimum = max(lab)
    while True:
        append_x = []
        append_y = []
        for x, y in zip(x_train_list, y_train_list):
            if lab[y] < minimum:
                append_x.append(x)
                append_y.append(y)
                lab[y] += 1
        x_train_list += append_x
        y_train_list += append_y
        if min(lab) >= minimum :
            break

    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list)
    
    return x_train, y_train

def cut_xy(x_train, y_train, minimum=None):
    
    cut_x = []
    cut_y = []
        
    if minimum == None:
        lab = [0 for _ in range(N)]
        for y in y_train:
            lab[y]+=1
        minimum = min(lab)
    
    lab2 = [0 for _ in range(N)]
    for x, y in zip(x_train , y_train):
        if lab2[y] > minimum:
            continue
        cut_x.append(x)
        cut_y.append(y)
        lab2[y] += 1
        
    return np.array(cut_x), np.array(cut_y)
        
def print_lab(ys):
    lab = [0 for _ in range(N)]
    for y in ys:
        lab[y] += 1
    print(lab)
           
def cut_by_length(xs, ys, length=0):
    '''
    left xs, ys which x's is longer than length
    '''
    cut_x = []
    cut_y = []
    for x,y in zip(xs, ys):
        if len(x) > length:
            cut_x.append(x)
            cut_y.append(y)
    
    return np.array(cut_x), np.array(cut_y)

def make_csr(xs, ys):
    ret_y = []
    ret_x = []
    i = 0
    a = len(ys)
    for x, y in zip(xs, ys):
        print(i, "/", a)
        tmp_x = texts_to_tlf_once_flat(x)
        '''
        if i == 0:
            np_x = np.array([tmp_x])
        else:
            np_x = np.append(np_x, [tmp_x], axis=0)
        '''
        ret_x += tmp_x
        ret_y.append(y)
        i += 1
    return ret_x, ret_y
        
def check_nan(xs):
    # xs is sparse matrix
    ret_x = []
    cur = 0
    maxi = len(xs)
    for x in xs:
        flag = 0
        print(cur, "/", maxi)
        x_a = x.toarray()
        for i in range(vocab_size):
            for j in range(N):
                if(np.isnan(x_a[i][j])==True):
                    print("It is nan.")
                    flag = 1
        cur += 1
        if flag == 0:
            ret_x.append([x])
    return ret_x
        


xs, ys = cut_by_length(xs, ys, length=50)
print_lab(ys)
x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.01, random_state=77, shuffle=True)
x_train, y_train = augment_xy(x_train, y_train, minimum=data_amount)
print_lab(y_train)
x_train, y_train = cut_xy(x_train, y_train, minimum=data_amount)
print_lab(y_train)
#x_train, y_train = make_csr(x_train, y_train)
#print_lab(y_train)

'''
try:
    with open('./data/x_train_tlf', 'wb') as f:
        pickle.dump(x_train, f)
    with open('./data/y_train_tlf', 'wb') as f:
        pickle.dump(y_train, f)
except:
    pass
'''
#print("Saving Complete")

#x_train = x_train.reshape(-1,vocab_size * N)
#print("SMOTE Processing....")
#smote = SMOTE(random_state=77)
#x_train, y_train = smote.fit_resample(x_train, y_train)
#print("SMOTE Finished")

#print_lab(y_train)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=77, shuffle=True)

'''
data augmentation
'''

tokenizer = Tokenizer(num_words=vocab_size)
#tokenizer = None
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_matrix(x_train, mode='tfidf')
x_valid = tokenizer.texts_to_matrix(x_valid, mode='tfidf')
x_test = tokenizer.texts_to_matrix(x_test, mode='tfidf')

#x_train = x_train.reshape(-1,vocab_size * N)
#x_train, y_train = smote.fit_resample(x_train, y_train)

Y_train = np.array(make_categorical(y_train))
Y_valid = np.array(make_categorical(y_valid))
Y_test = np.array(make_categorical(y_test))


train_generator = DataGenerator(x_train, Y_train, vocab_size=vocab_size, tokenizer=tokenizer, 
                                batch_size=batch_size, max_len=max_len, shuffle=True, mode='tfidf', smote=False)
valid_generator = DataGenerator(x_valid, Y_valid, vocab_size=vocab_size, tokenizer=tokenizer, 
                                batch_size=batch_size, max_len=max_len, shuffle=True, mode='tfidf', smote=False)
test_generator = DataGenerator(x_test, Y_test, vocab_size=vocab_size, tokenizer=tokenizer, 
                                batch_size=batch_size, max_len=max_len, shuffle=True, mode='tfidf')

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
#regularizer=tf.keras.regularizers.l2(lambd)
regularizer=None

txtInput = Input(shape=(vocab_size, ), name='txt_input')
#txtInput = Input(shape=(vocab_size, ), name='txt_input')
#txt = Embedding(vocab_size, embedding_dim, input_length=max_len)(txtInput)
#txt = GRU(256, return_sequences=True)(txtInput)
#txt = Dropout(dropout_rate)(txt)
#txt = GRU(128, return_sequences=True)(txt)
#txt = Dropout(dropout_rate)(txt)
#txt = GRU(64)(txt)
#txt = Dropout(dropout_rate)(txt)
#txt = GRU(64)(txt)
#txt = Dropout(dropout_rate)(txt)
#txt = GRU(512, return_sequences=True)(txt)
#txt = Dropout(dropout_rate)(txt)
#txt = GRU(512)(txt)
#txt = Dense(256, activation='relu', kernel_regularizer=regularizer)(txtInput)
#txt = Dropout(dropout_rate)(txt)
#txt = Dense(1024, activation='relu', kernel_regularizer=regularizer)(txtInput)
#txt = Dropout(dropout_rate)(txt)
#txt = Dense(256, activation='relu', kernel_regularizer=regularizer)(txt)
#txt = Dropout(dropout_rate)(txt)
#txt = Dense(64, activation='relu', kernel_regularizer=regularizer)(txt)
#txt = Dropout(dropout_rate)(txt)
#txt = Dense(32, activation='relu', kernel_regularizer=regularizer)(txt)
#txt = Dropout(dropout_rate)(txt)

#imgInput = Input(shape=(vocab_size, N, 1, ), name = 'img_input')
#img = [0 for _ in range(conv)]
#for i in range(conv):
#    img[i] = Conv2D(N, (1,N), padding='same', activation='swish')(imgInput)
#    img[i] = Conv2D(N, (1,N), padding='same', activation='swish')(img[i])
#    img[i] = MaxPooling2D((pool,1))(img[i])

#img = Conv2D(N, (1, N), padding='same', name='block1_conv1', activation='relu', kernel_regularizer=regularizer)(imgInput)
#img = Conv2D(N, (1, N), padding='same', name='block1_conv2', activation='relu', kernel_regularizer=regularizer)(img)
#img = Conv2D(N, (1,N), padding='same', name='block1_conv3', activation='relu')(img)
#img = MaxPooling2D((pool,1), name='block1_pool')(img)
#img = Conv2D(2 * N, (1,N), padding='same', name='block2_conv1', activation='relu', kernel_regularizer=regularizer)(img)
#img = Conv2D(2 * N, (1,N), padding='same', name='block2_conv2', activation='relu', kernel_regularizer=regularizer)(img)
#img = Conv2D(128, (3,3), padding='same', name='block2_conv3', activation='relu')(img)
#img = MaxPooling2D((pool,1), name='block2_pool')(img)
#img = Conv2D(4 * N, (1,N), padding='same', name='block3_conv1', activation='relu', kernel_regularizer=regularizer)(img)
#img = Conv2D(4 * N, (1,N), padding='same', name='block3_conv2', activation='relu', kernel_regularizer=regularizer)(img)#img = Conv2D(256, (3,3), padding='same', name='block3_conv3', activation='relu')(img)
#img = MaxPooling2D((pool,1), name='block3_pool')(img)
#img = Conv2D(8 * N, (1,N), padding='same', name='block4_conv1', activation='relu', kernel_regularizer=regularizer)(img)
#img = Conv2D(8 * N, (1,N), padding='same', name='block4_conv2', activation='relu', kernel_regularizer=regularizer)(img)
#img = MaxPooling2D((pool,1), name='block4_pool')(img)
#img = Conv2D(16 * N, (1, N), padding='same', name='block5_conv1', activation='relu', kernel_regularizer=regularizer)(img)
#img = Conv2D(16 * N, (1, N), padding='same', name='block5_conv2', activation='relu', kernel_regularizer=regularizer)(img)
#img = MaxPooling2D((pool,1), name='block5_pool')(img)
#img = Conv2D(32 * N, (1, N), padding='same', name='block6_conv1', activation='relu', kernel_regularizer=regularizer)(img)
#img = Conv2D(32 * N, (1, N), padding='same', name='block6_conv2', activation='relu', kernel_regularizer=regularizer)(img)
#img = MaxPooling2D((pool,1), name='block6_pool')(img)

#for i in range(conv):
#    img[i] = Flatten()(img[i])
    #img[i] = Dropout(dropout_rate)(img[i])

#output = Flatten()(img)
#img = Dense(512, activation='relu', kernel_regularizer=regularizer)(img)
#output = concatenate(img, axis=-1)

#output = Dropout(dropout_rate)(img)
#output = Dense(256, activation='relu', kernel_regularizer=regularizer)(output)
#output = Flatten()(imgInput)
output = Dense(128, activation='swish', kernel_regularizer=regularizer)(txtInput)
output = Dropout(dropout_rate)(output)
output = Dense(64, activation='swish', kernel_regularizer=regularizer)(output)
output = Dropout(dropout_rate)(output)
output = Dense(32, activation='swish', kernel_regularizer=regularizer)(output)
output = Dropout(0.2)(output)
output = Dense(N, activation='softmax')(output)

model = Model(inputs = txtInput, outputs = output)
#model = Model(inputs = imgInput, outputs = output)
#model = Model(inputs = [imgInput, txtInput], outputs = output)
sgd = SGD(learning_rate=cos_decay_ann, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
plot_model(model, to_file=model_name, show_shapes=True)
model.summary()
#model.load_weights('./checkpoint/epoch_img_1conv_1pool_030.ckpt')
#history = model.fit([X_train, X_train_txt], Y_train, validation_data=([X_valid, X_valid_txt], Y_valid), workers=-1, epochs = 1000, callbacks=[es, cl, tb, mc])
history = model.fit(x = train_generator, validation_data=valid_generator, shuffle=True, workers=-1, epochs = epoch, callbacks=[es, cl, tb, mc])

history_path = './history/' + m_n
with open(history_path, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.save(save_path)

test_loss, test_acc = model.evaluate(test_generator)

test_file = './logs/tests/' + m_n
cont = str(test_loss) + "\n" + str(test_acc)
with open(test_file, 'wb') as f:
    pickle.dump(cont, f)
