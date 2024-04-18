#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch


# In[2]:


from accelerate import Accelerator, DataLoaderConfiguration
from transformers import TrainingArguments
dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False, even_batches=True, use_seedable_sampler=True)
accelerator = Accelerator(dataloader_config=dataloader_config)


# In[3]:


import numpy as np
np.object = object
np.bool = bool
np.int = int


# In[ ]:


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)


data = pd.read_csv('matched_words_with_labels.csv', encoding='utf-8')

texts = data['Word'].tolist()
labels = data['Label'].tolist()


train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)


def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=150)

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)


train_dataset = Dataset.from_dict({"input_ids": train_encodings['input_ids'], "attention_mask": train_encodings['attention_mask'], "labels": train_labels})
val_dataset = Dataset.from_dict({"input_ids": val_encodings['input_ids'], "attention_mask": val_encodings['attention_mask'], "labels": val_labels})


model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)

# Set training parameters
training_args = TrainingArguments(
    output_dir='./results',                      # Output 
    evaluation_strategy="epoch",                 # Evaluated after each epoch
    learning_rate=2e-5,                          
    per_device_train_batch_size=16,             
    num_train_epochs=3,                          # Number of training rounds
    warmup_ratio=0.1,                            # Proportion of warm-up phase to total training steps
    lr_scheduler_type='linear',                  # Learning rate scheduler type
    logging_dir='./logs',                        
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)


trainer.train()

trainer.evaluate()

# Model is saved as 'my_model2'

model.save_pretrained('./my_model2')
tokenizer.save_pretrained('./my_model2')

