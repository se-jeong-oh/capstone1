class DataGenerator(Sequence):
  def __init__(self, X, tokenizer, vocab_size, batch_size, target_size = (224,224), shuffle = True):
    self.X = X
    self.length = len(X)
    self.batch_size = batch_size
    self.target_size = target_size
    self.shuffle = shuffle
    self.tokenizer = tokenizer
    self.vocab_size = vocab_size
    self.on_epoch_end()
  
  def on_epoch_end(self):
    self.indexes = np.arange(self.length)
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def __len__(self):
    return int(np.floor(self.length / self.batch_size))

  def __data_generation(self, i):  
    filepath = './images/' + self.X.iloc[i]['image']
    img = Image.open(filepath)
    img.draft('RGB', (224,224))
    img = img.resize(self.target_size, Image.NEAREST)
    img = np.array(img)
    img = img / 255.0

    tokenized = self.tokenizer.tokenize(self.X.iloc[i]['question'])
    question = self.tokenizer.convert_tokens_to_ids(tokenized)

    self.tokenizer.add_tokens(self.X.iloc[i]['answer'])
    answer = self.tokenizer.convert_tokens_to_ids(self.X.iloc[i]['answer'])

    return img, question, answer

  def __getitem__(self, index):
    img_data = np.empty((0,224,224,3))
    quest_data = []
    ans_data = []
    indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

    for i in indexes:
      img, question, answer = self.__data_generation(i)
      img_data = np.append(img_data, [img], axis=0)
      quest_data.append(question)
      ans_data.append(answer)

    quest_data = sequence.pad_sequences(quest_data, maxlen=max_len, padding='post')
    ans_data = to_categorical(ans_data, num_classes = self.vocab_size)

    return [quest_data, img_data], ans_data
