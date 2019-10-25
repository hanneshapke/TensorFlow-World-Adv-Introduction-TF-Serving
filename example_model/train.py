
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K


def load_training_data(fpath, num_val_samples=250):
    df = pd.read_csv(fpath, usecols=['SentimentText', 'Sentiment'])
    df = df.sample(frac=1).reset_index(drop=True)

    text = df['SentimentText'].tolist()
    text = [str(t).encode('ascii', 'replace') for t in text]
    text = np.array(text, dtype=object)[:]
    # text = np.array(text, dtype=object)[:, np.newaxis]
    # labels = np.asarray(pd.get_dummies(df.label), dtype=np.int8)
    labels = df['Sentiment'].tolist()
    labels = np.array(labels, dtype=int)[:]

    train_text = text[num_val_samples:]
    train_labels = labels[num_val_samples:]
    val_text = text[:num_val_samples]
    val_labels = labels[:num_val_samples]

    return (train_text, train_labels), (val_text, val_labels)


def get_model(num_categories=4):
    hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1", output_shape=[50],
                           input_shape=[], dtype=tf.string)

    # hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1", output_shape=[20],
    #                        input_shape=[], dtype=tf.string)

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='RMSProp', metrics=['acc'])
    return model


def train(fpath, epochs=4, batch_size=32):
    training_data, val_data = load_training_data(fpath)

    model = get_model()
    model.fit(training_data[0],
              training_data[1],
              validation_data=val_data,
              epochs=epochs,
              batch_size=batch_size)
    return model


def export_model(model, base_path="./exported_models/"):
    path = os.path.join(base_path, str(int(time.time())))
    tf.saved_model.save(model, path)


if __name__ == '__main__':
    fpath = "dataset.csv"
    model = train(fpath)
    export_model(model)