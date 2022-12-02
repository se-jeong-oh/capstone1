import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

with open('./data/tokenized_data.txt', 'rb') as lf:
    tok_org = pickle.load(lf)
tok = np.array(tok_org, dtype=object)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tok)

with open('./data/Y_train.txt', 'rb') as lf:
    Y_train = pickle.load(lf)
Y_train = np.array(Y_train, dtype=object)

Y_train_s = Y_train

Y_train_s = np.array(Y_train_s)

index_14 = []
for idx, y in enumerate(Y_train):
    if y[14] == 1:
        #print("Yes")
        index_14.append(idx)
#print(index_14)

index_9 = []
for idx, y in enumerate(Y_train):
    if y[9] == 1:
        #print("Yes")
        index_9.append(idx)
#print(index_9)

for idx in index_14:
    Y_train_s[idx][4] = 1
for idx in index_9:
    Y_train_s[idx][13] = 1
    
Y_train_s = np.delete(Y_train_s, [0,9,14], axis=1)
Y_train = Y_train_s
y_train = Y_train.tolist()

train_data = {}
y_train = Y_train.tolist()
train_data['sentence'] = tok
train_data['label'] = list(map(lambda idx: y_train[idx].index(max(y_train[idx])), range(len(y_train))))
train_df = pd.DataFrame(train_data)
train_df.to_pickle("./data/train_df.pkl")
train_load_df = pd.read_pickle("./data/train_df.pkl")

train_sent = train_load_df['sentence']
train_label = train_load_df['label']
x_valid, x_train, y_valid, y_train = train_test_split(train_sent, train_label, test_size=0.1, random_state=1, shuffle=True)

vocab_size = 50000
num_classes = 12
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(x_train)
X_train = tokenizer.texts_to_matrix(x_train, mode='tfidf')
X_valid = tokenizer.texts_to_matrix(x_valid, mode='tfidf')
Y_train = to_categorical(y_train, num_classes)
Y_valid = to_categorical(y_valid, num_classes)

print('훈련 샘플 본문의 크기 : {}'.format(X_train.shape))
print('훈련 샘플 레이블의 크기 : {}'.format(Y_train.shape))
print('테스트 샘플 본문의 크기 : {}'.format(X_valid.shape))
print('테스트 샘플 레이블의 크기 : {}'.format(Y_valid.shape))