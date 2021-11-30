import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.models import load_model


with open('data/small_vocab_en.txt') as f:
    english = f.readlines()

with open('data/small_vocab_fr.txt') as f:
    french = f.readlines()


class trainingPipeline:
    
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.length = max([len(sentence) for sentence in self.x])
        
    def tokenize(self,z):
        tok = Tokenizer()
        tok.fit_on_texts(z)
        return tok.texts_to_sequences(z), tok
    
    def padding(self,z,length=None):
        return pad_sequences(z, maxlen = self.length, padding = 'post')

    def preprocess(self):
        preprocess_x, self.x_tk = self.tokenize(self.x)
        preprocess_y, self.y_tk = self.tokenize(self.y)
        self.preprocess_x = self.padding(preprocess_x)
        preprocess_y = self.padding(preprocess_y)
        self.preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    
    def get_eng_pad(self):
        self.padded_english = self.padding(self.preprocess_x)

    def model_final(self,learning_rate):
        
        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(self.x_tk.word_index)+1,
                            output_dim=128,
                            input_length=self.padded_english.shape[1]))
        self.model.add(Bidirectional(GRU(256,
                                    return_sequences=False)))
        self.model.add(RepeatVector(self.preprocess_y.shape[1]))
        self.model.add(Bidirectional(GRU(256,return_sequences=True)))
        self.model.add(TimeDistributed(Dense(len(self.y_tk.word_index)+1,
                                        activation='softmax')))
        self.model.compile(loss = sparse_categorical_crossentropy, 
                     optimizer = Adam(learning_rate = learning_rate), 
                     metrics = ['accuracy'])
        
    def fit_model(self, b_size, epochs, val_split):
        self.model.fit(self.padded_english, self.preprocess_y, b_size, epochs,val_split)
        
    def save_the_model(self,path):
        self.model.save(path)
    
    def save_the_tokenizers(self,flag):
        if flag == 'eng':
            path = 'eng_tok.pickle'
            tokenizer = self.x_tk
        if flag == 'fr':
            path = 'fr_tok.pickle'
            tokenizer = self.y_tk
        with open(path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


tp = trainingPipeline(english, french)
tp.preprocess()
tp.get_eng_pad()
tp.model_final(0.005)
tp.fit_model(1024, 20, 0.2)
tp.save_the_model('data/my_model.h5')
tp.save_the_tokenizers('data/eng')
tp.save_the_tokenizers('data/fr')