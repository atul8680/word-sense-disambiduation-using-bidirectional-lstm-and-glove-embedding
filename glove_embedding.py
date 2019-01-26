import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class glove_embed:
    def __init__(self,input_data,EMBEDDING_FILE='glove.6B.100d.txt',vector_len=100):
        self.EMBEDDING_FILE = EMBEDDING_FILE
        self.vector_len = vector_len
        self.max_features=10000;
        self.tokenizer = Tokenizer(num_words= self.max_features)
        self.tokenizer.fit_on_texts(input_data)
        sequences = self.tokenizer.texts_to_sequences(input_data)
        self.data = pad_sequences(sequences, maxlen=100)
        
    def get_coefs(self,word,*arr):
        return word, np.asarray(arr, dtype='float32')
        
    def embedding(self):
        embeddings_index = dict(glove_embed.get_coefs(*o.strip().split()) for o in open(self.EMBEDDING_FILE,encoding="utf8"))
        all_embs = np.stack(embeddings_index.values())
        self.emb_mean = all_embs.mean()
        self.emb_std =  all_embs.std()
        
        self.word_index = self.tokenizer.word_index
        nb_words = min(self.max_features, len(self.word_index))
        self.embedding_matrix = np.random.normal(self.emb_mean, self.emb_std, (nb_words, self.vector_len))
        for word, i in self.word_index.items():
            if i >= self.max_features: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: self.embedding_matrix[i-1] = embedding_vector
        
        return self.word_index,self.embedding_matrix
            
    