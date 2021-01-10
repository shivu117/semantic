# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from flask import Flask, jsonify,render_template,request
#import tensorflow as tf

app = Flask(__name__)

path = '/content/gdrive/My Drive/sentiment/'

import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec


X_test = np.load(path+'X_test.npy')
vocab = np.load(path+'vocab.npy')
def word_to_num(dat):

  for i in range(len(dat)):
    dat[i] = '<BOS> '+ dat[i] +' <EOS>'
  temp1 = []
  for i in dat:
    temp = []
    for j in i.split():
      global vocab
      temp.append(vocab.index(j))
    temp1.append(temp)

  return np.array(temp1)







def create_embed(vocabs):
     word2vec = Word2Vec(vocabs,size=300)
     embeddings = np.random.randn(len(vocabs),300)
     for i in range(len(vocabs)):
       if vocabs[i] in word2vec.wv.vocab:
         embeddings[i] = word2vec.wv.word_vec(vocabs[i])
     return embeddings






def stack_LSTM_model(dt1, embeddings=embeddings, batch_size=16):
    layer2_1 = tf.keras.layers.Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings],
                                         batch_input_shape=[batch_size, None], trainable=False)

    layer2 = tf.keras.layers.LSTM(512, return_sequences=True, recurrent_initializer='glorot_uniform',
                                  recurrent_activation='sigmoid', stateful=True)
    layer3 = tf.keras.layers.LSTM(512, return_sequences=True, recurrent_initializer='glorot_uniform',
                                  recurrent_activation='sigmoid', stateful=True)
    layer4 = tf.keras.layers.Dense(150, activation='sigmoid')
    layer5 = tf.keras.layers.Dense(1, activation='sigmoid')

    layer1 = tf.keras.Input(shape=(None,), batch_size=batch_size)

    out1 = layer2_1(layer1)
    out1 = layer2(out1)
    out2 = layer3(out1)

    out3 = tf.keras.layers.GlobalMaxPooling1D()(out2)

    out3 = layer4(out3)

    out4 = layer5(out3)
    return tf.keras.models.Model(inputs=layer1, outputs=out4)


model = stack_LSTM_model(X_test[:20])
model.load_weight(path+'Stack_LSTM_model.h5')
# Press the green button in the gutter to run the script.


@app.route('/', methods=['POST'])
def Demo_page():

    temp = request.form['sentence']
    global vocab
    X = word_to_num(temp)
    predicted = model.predict(X)
    return render_template("demo_page.html", qlist = predicted)

@app.route('/')
def home():
    render_template("index.html")



if __name__=="__main__":
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
