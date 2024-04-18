This is the Final Project of DASC5420_01 - Theoretical Machine Learning  
Abstract 

This study explores the impact of news sentiment on stock price prediction using Natural Language Processing (NLP) techniques. Using a pre-trained RoBERTa model for sentiment scoring for financial news headlines and this is combined with technical factors for stock price prediction. Lasso and Ridge regression models are employed for prediction, with time-series cross-validation used for evaluation. Results show that sentiment scores significantly impact the prediction of certain stocks, such as FXI, but have less influence on others, like Tesla. While the models explain a substantial portion of stock price variation, further research is needed to enhance their robustness.

Keywords: Natural Language Processing, RoBERTa, Sentiment analysis, Stock price prediction, Lasso regression, Ridge regression, Time-series cross-validation



matched_words_with_labels.csv is the data got from dinancial dictionary and 
used to train the model to get model "my_model2"

selected_stock2.csv is the Reuters news data needs to be scored 

Lasso_Ridge_For_Tesla.py use data from tsla.csv to predict Tesla closing price
lasso_ridge_for_FXI.py use data from FXI.CSV to predict FXI closing price
