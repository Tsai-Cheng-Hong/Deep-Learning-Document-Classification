import sys
import os

import logging
import multiprocessing

import matplotlib.pyplot as plt
from gensim.models import Word2Vec

import pandas as pd
import sqlite3
import numpy as np
import re
import jieba

# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from __future__ import division,print_function,absolute_import
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, cross_validate, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# import xgboost as xgb
# import lightgbm as lgb
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
# from keras.utils import to_categorical
from tensorflow.python.keras import layers, models, optimizers
from tensorflow.python.keras.preprocessing import text, sequence
from tensorflow.python.keras.layers import Dense, Input, Flatten, Dropout, LSTM,\
    BatchNormalization,Bidirectional,TimeDistributed,SpatialDropout1D,\
    GlobalAveragePooling1D,ZeroPadding1D,Conv1D, MaxPooling1D, Embedding,Conv2D,MaxPooling2D,Concatenate,Multiply
from tensorflow.python.keras import Sequential
from tensorflow.python import keras
from tensorflow.python.keras import backend
from tensorflow.python.keras.callbacks import EarlyStopping
# 這邊記得替換成檔案的路徑
# import tensorflow as tf
import time

start_time = time.time()
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

df = pd.read_pickle('./data/undnews.pickle')

X_train, X_test, y_train, y_test = train_test_split(df['CONTENT_SEG'], df['SUB2'], test_size=0.2, random_state=42)
print("X_train.shape : ",X_train.shape)
print("X_test.shape : ", X_test.shape)
print("y_train.shape : ",y_train.shape)
print("y_test.shape : " ,y_test.shape)


#
class GCN(layers.Layer):
    def __init__(self, out_dim, kernel_size, kwarg_conv={}, kwargs_gate={}):
        super(GCN, self).__init__()
        self.conv = Conv1D(out_dim,kernel_size,**kwarg_conv)
        self.conv_gate = Conv1D(out_dim,kernel_size,activation='sigmoid',**kwargs_gate)
        self.pad_input = ZeroPadding1D(padding=(kernel_size-1,0))

    def call(self, inputs):
        X = self.pad_input(inputs)
        Y = self.conv(X)
        Z = self.conv_gate(X)
        return Multiply()([Y,Z])

# Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)




    # word2vec：詞向量是用一個多維度的向量來表示詞意的方法，用這個方法可以解決 CountVector/TFIDFVector
    # 在處理文本特徵的資料時出現非常龐大且稀疏的矩陣的問題。詞向量的取得方式有以下兩種
    # 運用我們這裡的資料自行訓練詞向量
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))
    # w2v_model = Word2Vec(df['CONTENT_SEG'].apply(lambda x: x.split(' ',-1)), # input要是list不是str
    #                      min_count=10,
    #                      size=200,
    #                      workers=multiprocessing.cpu_count())  # 訓練skip-gram模型
    # w2v_model = Word2Vec.load('./pretrained_model/wiki.zh.text.model')
# pretrain_w2v_model = Word2Vec.load('./pretrained_model/wiki.zh.text.model')
# pretrain_w2v_model = Word2Vec.load('./100D-ori/cbow.model')
# pretrain_w2v_model = Word2Vec.load('./word2vec.model')
pretrain_w2v_model = Word2Vec.load('./pretrained_model/wiki.zh.text.model')

print('model loading ok')

print('start training')
MAX_SEQUENCE_LENGTH = 300 # 每条新闻最大长度
EMBEDDING_DIM = 400 # 词向量空间维度

num_heads = 2
ff_dim = 32
    ## 超參數設定
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
MOMENTUM = 0.95
unit = 256
kernel_size = 3
stride = 1
    # 建立資料格式
# 10000 is 85.89 loss is 0.4208 Non-trainable params: 135,024,800
tokenizer = text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df['CONTENT_SEG'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
vec_train = tokenizer.texts_to_sequences(X_train)
vec_train = sequence.pad_sequences(vec_train, maxlen=MAX_SEQUENCE_LENGTH)
vec_test = tokenizer.texts_to_sequences(X_test)
vec_test = sequence.pad_sequences(vec_test, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of vec_train tensor:', vec_train.shape)
print('Shape of vec_test tensor:', vec_test.shape)

y_train_dummy = to_categorical(np.asarray(y_train))
y_test_dummy = to_categorical(np.asarray(y_test))
print('Shape of y_train tensor:', y_train_dummy.shape)
print('Shape of y_test tensor:', y_test_dummy.shape)
    # 將詞替換成對應的向量
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in pretrain_w2v_model:
        embedding_matrix[i] = np.asarray(pretrain_w2v_model[word],
                                            dtype='float32')



embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    # 搭建模型框架

backend.clear_session()
model = Sequential()
model.add(embedding_layer)

# CNN
model.add(BatchNormalization())
# model.add(Activattion('relu'))
# model.add(Conv1D(EMBEDDING_DIM, 3, padding='valid', activation='relu', strides=1))
# model.add(MaxPooling1D(pool_size=2,strides=1))
#Gated Convolution
Gated_CNN = GCN(out_dim=EMBEDDING_DIM,kernel_size=kernel_size)
model.add(Gated_CNN)
model.add(Dropout(0.2))
Gated_CNN = GCN(out_dim=200,kernel_size=kernel_size)
model.add(Gated_CNN)
model.add(Dropout(0.2))
Gated_CNN = GCN(out_dim=100,kernel_size=kernel_size)
model.add(Gated_CNN)
model.add(Dropout(0.2))
#Transformer
# transformer_block = TransformerBlock(embed_dim=EMBEDDING_DIM,num_heads=num_heads,ff_dim=ff_dim)
# model.add(transformer_block)
# model.add(GlobalAveragePooling1D())
# model.add(Dropout(0.1))
# model.add(Dense(20,activation=("relu")))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
# model.add(transformer_block)
# model.add(BatchNormalization())
# model.add(transformer_block)
# model.add(BatchNormalization())
# model.add(transformer_block)
# model.add(MaxPooling1D(pool_size=2,strides=1))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(Flatten())
# model.add(Dense(EMBEDDING_DIM, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(EMBEDDING_DIM, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(y_train_dummy.shape[1], activation='softmax'))

# Bi-LSTM
# model.add(BatchNormalization())
# model.add(Bidirectional(LSTM(200,return_sequences=True,dropout=0.2, recurrent_dropout=0.2,activation = 'tanh')))
# model.add(Bidirectional(LSTM(200,return_sequences=False, dropout=0.2, recurrent_dropout=0.2,activation = 'tanh')))
# # model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2,activation = 'tanh'))
# model.add(BatchNormalization())
# model.add(Dense(EMBEDDING_DIM,activation='relu')) #no use is 83.04
# model.add(BatchNormalization())
# model.add(Dropout(0.2))
# model.add(Dense(y_train_dummy.shape[1], activation='softmax'))

# model.add(SpatialDropout1D(0.2)) #this drop is for nlp model
#
# #原本的LSTM
# model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2,activation = 'tanh'))
# model.add(BatchNormalization())
# model.add(Dense(y_train_dummy.shape[1], activation='softmax'))
#
# model.summary()
    # 載入 Callbacks, 並將 monitor 設定為監控 validation loss
earlystop = EarlyStopping(monitor="val_loss",
                            patience=5,
                            verbose=1,
                          restore_best_weights=True)
# earlystop = EarlyStopping(monitor="val_acc",
#                             patience=5,
#                             verbose=1)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, decay=0.0)
model.compile(loss='categorical_crossentropy',  optimizer='adam'  , metrics=['accuracy'])
# model.compile(loss='spare_categorical_crossentropy',  optimizer='adam'  , metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy',
    #             optimizer= Adam(lr=LEARNING_RATE, epsilon=None, decay=0.0),
    #             metrics=['accuracy'])

history = model.fit(vec_train, y_train_dummy,
            epochs=EPOCHS,
            validation_split=0.2,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=[earlystop])


#
# model.fit(vec_train, y_train_dummy,
#             epochs=EPOCHS,
#             validation_split=0.2,
#             batch_size=BATCH_SIZE,
#             shuffle=True)

model.evaluate(vec_test, y_test_dummy,batch_size=BATCH_SIZE)
model.summary()

end_time = time.time()
training_time = end_time - start_time
print("cost time : ",training_time," sec")
# Loss
plt.title('Loss')
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.legend()
plt.show()
# Accuracy
plt.title('Accuracy')
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='test')
plt.legend()
plt.show()
print('ok')
# First time : 83.19 no restore the best model & weight
# Second time : 84.87 restore the best model & weight
# Third time : 85.08 double Bi-LSTM
# Forth time :  84.4 single Bi-LSTM
# Fifth time :  85.02 single LSTM
# sixth time :  85.02 double LSTM
# seven time :  84.22 many to many LSTM
# eigth time :  84.06 many to many LSTM (No use timeDistributed)
