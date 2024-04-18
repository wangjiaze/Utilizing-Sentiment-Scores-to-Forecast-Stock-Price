#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from textblob import TextBlob,Word
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases, Phraser
import csv
import nltk
# nltk.download('wordnet')


# In[2]:


def get_tweet_sentiment(text):
  print(text)
  return text


# In[3]:


test=pd.read_csv("FIN111K_selected.csv")
# 1. Convert to lowercase
test['Headline']=test['Headline'].apply(lambda x:" ".join([word.lower() for word in x.split()]))

# 2. Remove punctuation
test['Headline']=test['Headline'].str.replace('[^\w\s]','')

# 3. Remove stopwords
stop=stopwords.words('english')
test['Headline']=test['Headline'].apply(lambda x:" ".join(word for word in x.split() if word not in stop))

# 4. Remove frequent words
freq=pd.Series(' '.join(test['Headline']).split()).value_counts()[:10]
test['Headline']=test['Headline'].apply(lambda x:" ".join(word for word in x .split() if word not in freq))

# 5. Remove rare words
rare=pd.Series(' '.join(test['Headline']).split()).value_counts()[-10:]
test['Headline']=test['Headline'].apply(lambda x:" ".join(word for word in x.split() if word not in rare))

# 6. Spelling correction
test['Headline']=test['Headline'].apply(lambda x: str(TextBlob(x).correct()))

# 7. Lemmatization
test['Headline']=test['Headline'].apply(lambda x:" ".join([Word(word).lemmatize() for word in x.split()]))
print(test['Headline'].head(10))

# Tokenization using NLTK
tokenized_texts = [word_tokenize(text) for text in test['Headline']]

# Phrase detection using gensim
phrases = Phrases(tokenized_texts, min_count=1, threshold=1)
phraser = Phraser(phrases)

# Apply detected phrases to original text
phrased_texts = [phraser[text] for text in tokenized_texts]
file = open('word_splited.csv', 'w', encoding='utf-8')
writer = csv.writer(file)
writer.writerows(phrased_texts)
file.close()


# In[ ]:


from collections import Counter

file_path = 'word_splited.csv'
df = pd.read_csv(file_path, sep="\t", header=None, names=['text'])
df.head()

words = df['text'].str.split(',').explode()
word_counts = Counter(words)

word_counts_df = pd.DataFrame(word_counts.items(), columns=['word', 'number'])

output_file_path = 'word_counted.csv'
word_counts_df.to_csv(output_file_path, index=False)

output_file_path


# In[ ]:


import pandas as pd
import csv


text = pd.read_csv("word_counted.csv")
# print(text['word'].head(10))

# Remove numbers from text
text['word'] = text['word'].apply(lambda x: ''.join(filter(lambda c: not c.isdigit(), str(x))))

# finance dictionary
finance_dict = pd.read_csv("Loughran-McDonald_MasterDictionary_1993-2021.csv")
# print(finance_dict['Word'].head(10))

# Convert words in finance dictionary to lowercase
finance_dict['Word'] = finance_dict['Word'].str.lower()

# Get set of words from text
set1 = set(text['word'])

# Get set of words from finance dictionary
set2 = set(finance_dict['Word'])

# Match words from finance dictionary
matching_words = list(set1.intersection(set2))
# print("Matching words:", matching_words)

# Calculate match rate
match_rate = (len(matching_words) / len(set1)) * 100
print("Match rate: {:.2f}%".format(match_rate))
# Write to CSV file
with open('mat.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for word in matching_words:
        writer.writerow([word])


# In[ ]:


# Get the labeled words

csv_file_path = 'Loughran-McDonald_MasterDictionary_1993-2021.csv'

df_words = pd.read_csv(csv_file_path)
def determine_label(row):
    if row['Positive'] > 0:
        return 2
    elif row['Negative'] > 0:
        return 0
    else:
        return 1

df_words['Label'] = df_words.apply(determine_label, axis=1)

df_words[['Word', 'Positive', 'Negative', 'Label']].head(9)

uploaded_csv_path = 'mat.csv'
rows = []
with open(uploaded_csv_path, 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    rows = [row for row in csvreader]
if rows:  
    if rows[0]:  
        if any(not cell.replace('.', '', 1).isdigit() for cell in rows[0]):
            rows[0][0] = 'words'
        else:
            rows.insert(0, ['words'] + [""] * (len(rows[0]) - 1))
else:
    rows.append(['words'])
modified_csv_path = 'modified_mat.csv'
with open(modified_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)

modified_csv_path
df_new_to_match = pd.read_csv(modified_csv_path)

df_new_to_match.columns.tolist()

df_new_to_match.rename(columns={'words': 'Word'}, inplace=True)

df_new_to_match['Word'] = df_new_to_match['Word'].str.upper()

df_new_matched = pd.merge(df_new_to_match, df_words[['Word', 'Label']], on='Word', how='inner')

new_matched_output_file_path = 'matched_words_with_labels.csv'
df_new_matched.to_csv(new_matched_output_file_path, index=False)

new_matched_output_file_path

