import pandas as pd
import pickle

'''
기존에 계산했던 TLF 에서 TF도 고려
해당 label에서 해당 단어가 몇번 등장했는지
TF = 특정 label에 속한 해당 단어의 개수 / 특정 label에 속한 단어의 개수 
'''
N = 12
#vocab_size = 50000

train_load_df = pd.read_pickle("./data/train_df.pkl")
#tlf_dict = pd.read_pickle("./data/tlf_dict")
x_train = train_load_df['sentence']
y_train = train_load_df['label']
#ttokenizer = Tokenizer(num_words = vocab_size)
#ttokenizer.fit_on_texts(x_train)

tlf_dict = pd.read_pickle("./data/tlf_dict_2")

def find_word(word, label):
    count = 0
    for x, y in zip(x_train, y_train):
        if y != label:
            continue
        else:
            for words in x:
                if words == word:
                    count += 1
    return count

for word, values in tlf_dict.items():
    print("Processing, ", word)
    for i in range(N):
        tlf_dict[word][i] *= find_word(word, i)

with open('./data/tlf_dict_3', 'wb') as f:
    pickle.dump(tlf_dict, f)