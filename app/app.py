import pickle
import logging
import tensorflow as tf
import numpy as np
from ast import literal_eval
from flask import Flask, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
graph = tf.get_default_graph()
sess = tf.Session()
set_session(sess)

@app.route('/predict',methods=['POST'])
def predict():
	with graph.as_default():
		set_session(sess)
		response = literal_eval(request.data.decode('utf8'))
		sentence = response['message']
		y_id_to_word = {value: key for key, value in fr_tok.word_index.items()}
		y_id_to_word[0] = '<PAD>'
		sentence = [eng_tok.word_index[word] for word in sentence.split()]
		sentence = pad_sequences([sentence], maxlen=15, padding='post')
		pred = model.predict(sentence)
		final = ' '.join([y_id_to_word[np.argmax(x)] for x in pred[0]])
		final = final.replace('<PAD>',' ')
		return final

if __name__ == '__main__':
	model = load_model('../data/my_model.h5')
	with open('../data/fr_tok.pickle', 'rb') as handle:
		fr_tok = pickle.load(handle)
	with open('../data/eng_tok.pickle', 'rb') as handle:
		eng_tok = pickle.load(handle)
	with open('../data/small_vocab_en.txt') as f:
		english = f.readlines()
	logging.info('Model and vectorisers loaded')
	logging.info('App starting')
	app.run(debug=True)