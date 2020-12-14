import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import json, os, random
from tensorflow.python.keras.models import model_from_json
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import html2text
import requests

def load_directory_data(train_path):
  data = {}
  data["sentence"] = []
  for fname in sorted(os.listdir(train_path)):
    fdata = os.path.join(train_path, fname, 'news content.json')
    if os.path.isfile(fdata):
      with open(fdata) as f:
        story = json.loads(f.read())
        data["sentence"].append(story['text'])
  return pd.DataFrame.from_dict(data)

def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "real"))
  neg_df = load_directory_data(os.path.join(directory, "fake"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

def download_and_load_datasets(force_download=False):
  data_path = '/home/user/fnn/politifact/'
  train_df = load_dataset(data_path)[:875]
  test_df = load_dataset(data_path)[-100:]
  return train_df, test_df

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

train_df, test_df = download_and_load_datasets()
train_df.head()
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["polarity"], shuffle=False)

embedded_text_feature_column = hub.text_embedding_column(
    key="sentence", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.estimator.DNNClassifier(
    model_dir='/home/user/fnnmodels',
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

estimator.train(input_fn=train_input_fn, steps=1000)
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))
