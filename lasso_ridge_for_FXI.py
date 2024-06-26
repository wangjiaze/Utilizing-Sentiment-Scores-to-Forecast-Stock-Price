import pandas as pd
import ast
import talib
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Read the CSV file
df = pd.read_csv(r'selected_stock_cutted2.csv')
df['date'] = pd.to_datetime(df['date'], utc=True).dt.date

# Convert the 'sentiment_scores' column to DataFrame columns
sentiment_scores_df = pd.json_normalize(df['sentiment_scores'].apply(ast.literal_eval))
df = pd.concat([df, sentiment_scores_df], axis=1)
df['score'] = ((df['positive']-df['negative']) / (df['negative'] + df['positive'])) * (1 - df['neutral'])

# Get unique stock codes
stocks = df['stock'].unique()

# Split data by stock code and save to different files
for stock in stocks:
    stock_data = df[df['stock'] == stock]
    mean_scores_stock = stock_data.groupby('date')['score'].max().reset_index()
    if stock == 'FXI':
        mean_scores_FXI = mean_scores_stock

# Read stock data
stock = pd.read_csv(r'FXI.csv')
stock = stock.dropna()
stock['Close*'] = pd.to_numeric(stock['Close*'], errors='coerce')
stock['Open'] = pd.to_numeric(stock['Open'], errors='coerce')
stock['chg'] = (stock['Close*'] - stock['Open'])
stock['pct-chg'] = (stock['Close*'] - stock['Open']) / stock['Open']
stock['sma'] = talib.SMA(stock['Close*'], timeperiod=5)
stock['rsi'] = talib.RSI(stock['Close*'], timeperiod=14)
stock['macd'], stock['signal'], stock['_'] = talib.MACD(stock['Close*'], fastperiod=6, slowperiod=13, signalperiod=5)
stock['upper_band'], middle_band, stock['lower_band'] = talib.BBANDS(stock['Close*'], timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
stock = stock.dropna()
stock['Date'] = pd.to_datetime(stock['Date'])
mean_scores_FXI.reset_index(inplace=True)
mean_scores_FXI['date'] = pd.to_datetime(mean_scores_FXI['date'])
merged_data_tsla = pd.merge(stock, mean_scores_FXI, left_on='Date', right_on='date', how='left')
merged_data_tsla.drop('date', axis=1, inplace=True)

# Scatter plot of sentiment score vs price change
plt.figure()
plt.scatter(merged_data_tsla['score'], merged_data_tsla['pct-chg'])
plt.xlabel('Sentiment Score')
plt.ylabel('Price Change (%)')
plt.title('Relationship between Sentiment Score and Price Change For FXI')
plt.show()

# Lasso Regression
X = merged_data_tsla[['sma', 'rsi', 'macd', 'signal', 'index', 'score', 'upper_band', 'lower_band']]
y = merged_data_tsla['Close*']
lasso = Lasso()
param_grid = {'alpha': np.arange(0.00001, 100, 1).tolist()}
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)
print("Best alpha:", grid_search.best_params_)
print("Best MSE:", -grid_search.best_score_)
lasso = Lasso(alpha=0.0001)
tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []
r2_scores = []
preds = []
actuals = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    preds.extend(y_pred)
    actuals.extend(y_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
average_mse = sum(mse_scores) / len(mse_scores)
average_r2 = sum(r2_scores) / len(r2_scores)
preds_with_sentiment_score = preds
print('LASSO')
print(f"Average MSE: {average_mse}")
print(f"Average R²: {average_r2}")

# Feature Importance in Lasso Model
feature_importance = lasso.coef_
for i, feature in enumerate(['sma', 'rsi', 'macd', 'signal', 'index', 'score', 'upper_band', 'lower_band']):
    print(f"{feature}: {feature_importance[i]}")
plt.figure(figsize=(12, 6))
plt.bar(['sma', 'rsi', 'macd', 'signal', 'index', 'score', 'upper_band', 'lower_band'], feature_importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance in Lasso Model For FXI')
plt.show()

# Lasso Regression without sentiment score
print('Lasso without sentiment score')
X = merged_data_tsla[['sma', 'rsi', 'macd', 'signal', 'index', 'upper_band', 'lower_band']]
y = merged_data_tsla['Close*']
lasso = Lasso()
param_grid = {'alpha': np.arange(0.00001, 100, 1).tolist()}
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)
print("Best alpha:", grid_search.best_params_)
print("Best MSE:", -grid_search.best_score_)
lasso = Lasso(alpha=0.0001)
tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []
r2_scores = []
preds = []
actuals = []
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    preds.extend(y_pred)
    actuals.extend(y_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)
average_mse = sum(mse_scores) / len(mse_scores)
average_r2 = sum(r2_scores) / len(r2_scores)
print('LASSO')
print(f"Average MSE: {average_mse}")
print(f"Average R²: {average_r2}")

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(preds)), preds_with_sentiment_score, label='Predicted_with_sentiment_score', linestyle='--')
plt.plot(np.arange(len(preds)), preds, label='Predicted_without_sentiment_score', linestyle='-.')
plt.plot(np.arange(len(actuals)), actuals, label='Actual', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Lasso Predicted vs Actual Values For Fxi')
plt.legend()
plt.show()

# Ridge Regression
X = merged_data_tsla[['sma', 'rsi', 'macd', 'signal', 'index', 'score', 'upper_band', 'lower_band']]
y = merged_data_tsla['Close*']
alpha_values = np.arange(0.00001, 100, 1).tolist()
alpha_results = []
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []
    r2_scores = []
    preds = []
    actuals = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        preds.extend(y_pred)
        actuals.extend(y_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
    average_mse = sum(mse_scores) / len(mse_scores)
    average_r2 = sum(r2_scores) / len(r2_scores)
    alpha_results.append((alpha, average_r2))

results_df = pd.DataFrame(alpha_results, columns=['Alpha', 'Average R²'])
preds_with_sentiment_score = preds

plt.figure(figsize=(10, 6))
plt.plot(results_df['Alpha'], results_df['Average R²'], marker='o')
plt.xlabel('Alpha')
plt.ylabel('Average R²')
plt.title('Average R² vs Alpha for Ridge Regression For FXI')
plt.xscale('log')
plt.grid(True)
plt.show()
print('ridge regression ')
print(max(results_df['Average R²']))

# Ridge Regression without sentiment score
X = merged_data_tsla[['sma', 'rsi', 'macd', 'signal', 'index', 'upper_band', 'lower_band']]
y = merged_data_tsla['Close*']
alpha_values = np.arange(0.00001, 100, 1).tolist()
alpha_results = []
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []
    r2_scores = []
    preds = []
    actuals = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        ridge.fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        preds.extend(y_pred)
        actuals.extend(y_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
    average_mse = sum(mse_scores) / len(mse_scores)
    average_r2 = sum(r2_scores) / len(r2_scores)
    alpha_results.append((alpha, average_r2))

results_df = pd.DataFrame(alpha_results, columns=['Alpha', 'Average R²'])
print('without sentiment score, the R2 is')
print(max(results_df['Average R²']))

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(preds)), preds_with_sentiment_score, label='Predicted_with_sentiment_score', linestyle='--')
plt.plot(np.arange(len(preds)), preds, label='Predicted_without_sentiment_score', linestyle='-.')
plt.plot(np.arange(len(actuals)), actuals, label='Actual', linestyle='-')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Ridge Predicted vs Actual Values For Fxi')
plt.legend()
plt.show()







