import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

vocab_size = 50000
N = 12
ilf_dict = {}

train_load_df = pd.read_pickle("./data/train_df.pkl")
#tlf_dict = pd.read_pickle("./data/tlf_dict")
x_train = train_load_df['sentence']
y_train = train_load_df['label']

ttokenizer = Tokenizer(num_words = vocab_size)
ttokenizer.fit_on_texts(x_train)
X_train = ttokenizer.texts_to_sequences(x_train)

word_dic = ttokenizer.index_word

lab = [0 for _ in range(N)]
for y in y_train:
    lab[y] += 1
cur = 0
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
    print("Preprocessing : ", cur, "/", 29358)
    cur += 1
try:
    with open('./data/tlf_dict_2', 'wb') as fp:
        pickle.dump(ilf_dict, fp)
except:
    pass