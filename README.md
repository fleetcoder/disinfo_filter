# disinfo_filter
## a TensorFlow app trained on the FakeNewsNet data set

## to develop your own disinformation filter, start by downloading FakeNewsNet:

https://github.com/KaiDMML/FakeNewsNet

## use the train_tensorflow.py app from this repository to train a model

edit the file named train_tensorflow.py:

- change the model-output-directory "/home/user/fnnmodels" 

- change the data-input-directory "/home/user/fnn/politifact/" 

install dependencies:

  pip3 install tensorflow, tensorflow_hub, matplotlib, numpy, pandas, seaborn, html2text, requests

run:

  python3 train_tensorflow.py
  
## Classify a news article as disinformation or not with the filter:

(change "/home/user/fnnmodels" to location of trained model)

(example.py)

```
embedded_text_feature_column = hub.text_embedding_column(
  key="sentence",
  module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

estimator = tf.estimator.DNNClassifier(
  model_dir="/home/user/fnnmodels",
  hidden_units=[500, 100],
  feature_columns=[embedded_text_feature_column],
  n_classes=2,
  optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

text_of_news_story = "here is the body of a news article"
data = {}
data["sentence"] = []
data["sentence"].append(text_of_news_story)
neg_df = pd.DataFrame.from_dict(data)
test_df = pd.concat([neg_df]).sample(frac=1).reset_index(drop=True)
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn( test_df, shuffle=False )
test_predict_generator = estimator.predict( input_fn=predict_test_input_fn )
output = 0
for res in test_predict_generator:
  output = int(res['class_ids'][0])
print(str(output))
```


and run:
  python3 example.py

output: 1 - content resembles disinformation

output: 0 - does not trip the filter

  
  
