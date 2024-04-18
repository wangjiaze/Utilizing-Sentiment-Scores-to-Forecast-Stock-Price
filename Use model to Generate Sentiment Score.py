#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import transformers
transformers.trainer.np.object = object
from transformers import Trainer
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoConfig
from scipy.special import softmax


# In[7]:


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Load model 'my_model2'

MODEL = f'my_model2/'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_sentiment_scores(text):
    processed_text = preprocess(text)
    encoded_input = tokenizer(processed_text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sentiment_scores = {config.id2label[i]: score for i, score in enumerate(scores)}
    return sentiment_scores

file_path = 'selected_stock2.csv'
try:
    # Try to read the file using UTF-8 encoding
    data_to_analyze = pd.read_csv(file_path, encoding='utf-8')  
except UnicodeDecodeError:
    
    # If that fails, try to read the file using ISO-8859-1 encoding
    data_to_analyze = pd.read_csv(file_path, encoding='ISO-8859-1')  

# Store sentiment scores
data_to_analyze['sentiment_scores'] = data_to_analyze['title'].apply(lambda x: get_sentiment_scores(x))

data_to_analyze.to_csv('selected_stock_cutted2.csv', index=False)

