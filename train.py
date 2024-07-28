import streamlit as st
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import zipfile


nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def train_chatbot(intents, learning_rate, epochs, batch_size):
    words = []
    classes = []
    documents = []
    ignore_letters = ["?", "!", ",", ".", "'"]

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent["tag"])

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append(bag + output_row)

    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = np.array([np.array(x[:len(words)], dtype=np.float32) for x in training])
    train_y = np.array([np.array(x[len(words):], dtype=np.float32) for x in training])

    #creating the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(train_y[0]),activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    t = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save("chatbot_model.h5",t)

    return "Model trained successfully. Files 'words.pkl', 'classes.pkl', 'intents.json' and 'chatbot_model.h5' have been created."