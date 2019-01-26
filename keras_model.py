import keras
import tensorflow as tf
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Bidirectional
from keras.models import Model
from keras.utils import to_categorical
from attention_decoder import Attention
from glove_embedding import glove_embed
from preprocessing import Preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Keras_model:
    def __init__(self,out_classes,vector_len=100):
        preprocess=Preprocessing()
        data1=preprocess.clean_text('phone2.csv')
        data2=preprocess.clean_text('product2.csv')
        data3=preprocess.clean_text('line_cord2.csv')
        data4=preprocess.clean_text('division2.csv')
        data5=preprocess.clean_text('formation2.csv')
        
        out_lable=429*[0] +2218*[1] + 373*[2]+ 376*[3] + 349*[4]
        input_data=data1+data2+data3+data4+data5
        self.cat_labels=to_categorical(out_lable,num_classes=5)
        
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(input_data)
        sequences = tokenizer.texts_to_sequences(input_data)
        self.data = pad_sequences(sequences, maxlen=100)

        self.out_classes=out_classes
        self.vector_len=vector_len;
        glove=glove_embed( input_data)
        self.word_index,self.embedding_matrix=glove.embedding() #def rnn_model(self):
        
        inp = Input(shape=(self.vector_len,))
        x = Embedding(10000, self.vector_len, weights=[self.embedding_matrix])(inp)
        x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.25)(x)
        x = Attention()(x)
        x = Dense(5, activation="softmax")(x)
        self.model = Model(inputs=inp, outputs=x)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    def run_mod(self):
        self.model.fit(self.data, self.cat_labels,validation_split=0.3, batch_size=100, epochs=10)
        