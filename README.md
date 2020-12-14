# disinfo_filter
## a TensorFlow app trained on the FakeNewsNet data set

## to develop your own disinformation filter, start by downloading FakeNewsNet:

https://github.com/KaiDMML/FakeNewsNet

## use the train_tensorflow.py app from this repository to train a model

change the model-output-directory "/home/user/fnnmodels" 

change the data-input-directory "/home/user/fnn/politifact/" 

install dependencies:

  pip3 install tensorflow, tensorflow_hub, matplotlib, numpy, pandas, seaborn, html2text, requests

then run:

  python3 train_tensorflow.py
  
## Classify a news article as disinformation with the filter:

(example.py)

```text_of_news_story = "here is the body of a news article"
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
```


and run:
  python3 example.py
  
  
  
