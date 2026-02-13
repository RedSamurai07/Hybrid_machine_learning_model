# Hybrid Machine Learning Model

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview

The financial market is highly volatile, and capturing both long-term trends and short-term sequential dependencies is crucial for accurate stock price prediction. This project aims to build a Hybrid Machine Learning Model by combining LSTM (Long Short-Term Memory) and Linear Regression to leverage their unique predictive strengths. The goal is to provide a robust forecasting system that outperforms single-algorithm approaches by capturing complex data patterns and broader market trends simultaneously.

### Executive Summary

1. Data Science Team: This analysis equips the technical team with a sophisticated architecture that handles time-series data more effectively. By scaling data with MinMaxScaler and using 60-day sequences, the team can stabilize training and improve model convergence.

2. Financial Analysts: For analysts, the hybrid model offers a dual-perspective forecast. It provides insights into immediate price movements (via LSTM) while maintaining a grounded view of the overall price trajectory (via Linear Regression), reducing the risk of "overfitting" to market noise.

3. Strategic Stakeholders: Decision-makers can leverage this model to gain higher confidence in price predictions. The hybrid approach mitigates the weaknesses of individual models, providing a more reliable foundation for high-stakes investment strategies and risk management.

### Goal

The objective of this analysis is to:

1. Capture Sequential Dependencies – Use LSTM to identify complex patterns in historical 60-day price sequences.

2. Identify Linear Trends – Utilize Linear Regression to capture the overarching direction of the data.

3. Improve Predictive Robustness – Combine model outputs to reduce error metrics (MSE/RMSE) compared to standalone models.

4. Optimize Data Pre-processing – Implement feature scaling and windowing techniques to ensure data compatibility across different algorithm types.

5. Develop a Scalable Framework – Create a repeatable Python-based workflow for hybrid modeling that can be applied to various time-series datasets.

### Data structure and initial checks
[Dataset](https://docs.google.com/spreadsheets/d/1_SPTTQvCc3E5TKyG4ZY4ZLLXpK-thUsrra5aEo0CntM/edit?usp=sharing)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 


### Tools
- Excel : Google Sheets - Check for data types, Table formatting
- Python: Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, inferential Statistics, Data manipulation and Analysis(Numpy, Pandas),Visualization (Matplotlib, Seaborn), Feature Engineering, Hypothesis Testing
  
### Analysis
1). Python
Importing all the libraries
```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
```
```python
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
```
Loading the dataset
```python 
data = pd.read_csv('apple_stock_data.csv')
data.head()
```
<img width="622" height="219" alt="image" src="https://github.com/user-attachments/assets/32be4da4-49eb-43eb-80a6-938bedaf636f" />

Checked the dimensions & shape of the data and let's look into the information about the dataset.
``` python
data.info()
```
<img width="408" height="320" alt="image" src="https://github.com/user-attachments/assets/9093d15f-0b13-4034-9280-bf3526745e79" />

Data Preprocessing

Handling Null values in our data.
``` python
data.isna().sum()
```

<img width="140" height="154" alt="image" src="https://github.com/user-attachments/assets/8bb6cf21-3e4d-42ff-98f6-bbf365727a32" />

We notice that one of the features is not spaced looks suspicious for an extra space. let's check for the complete features.
``` python
data.columns
```

Data types converison need to be done on the date feature.
``` python
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
```
We need to rename most of our features and convert their datatypes to numeric format
``` python
data['Close'] = data[' Close/Last'].rename('Close',inplace = True)
data['Volume'] = data[' Volume'].rename('Volume',inplace = True)
data['Open'] = data[' Open'].rename('Open',inplace = True)
data['High'] = data[' High'].rename('High',inplace = True)
data['Low'] = data[' Low'].rename('Low',inplace = True)
```
``` python

data = data.drop(columns=[' High', ' Low', ' Open', ' Close/Last', ' Volume'])
data['Close'] = pd.to_numeric(data['Close'].astype(str).str.replace('$', '').str.strip(), errors='coerce')
data['High'] = pd.to_numeric(data['High'].astype(str).str.replace('$', '').str.strip(), errors='coerce')
data['Low'] = pd.to_numeric(data['Low'].astype(str).str.replace('$', '').str.strip(), errors='coerce')
data['Open'] = pd.to_numeric(data['Open'].astype(str).str.replace('$', '').str.strip(), errors='coerce')

data = data.dropna()

df = data.reset_index()
```
Now our data is cleaned & looks tidy

<img width="581" height="228" alt="image" src="https://github.com/user-attachments/assets/3fe00677-5618-44ad-8ffc-40df00d06e1b" />

Exploratpry Data Analysis
``` python
def plots(data,features = ['Open','Low','High','Close']):
    plt.figure(figsize=(10, 8))
    for feature in features:
        plt.plot(data['Date'], data[feature], label=f'{feature} Price')
        plt.title('Apple Stock Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid()
        plt.show()

plots(df)
```
<img width="850" height="701" alt="image" src="https://github.com/user-attachments/assets/cd743575-b860-4b22-af64-ceb50596ef08" />

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/5cb4af83-efc8-4184-a52a-e70f52ac7f56" />

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/ba4c7c57-0901-4312-9db6-5751fed09d5e" />

<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/d431e1bd-a988-4976-bcff-7cd329263bd7" />

``` python
plt.plot(df['Date'], df['Volume'], label='Volume')
plt.title('Apple Stock Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
```
<img width="554" height="455" alt="image" src="https://github.com/user-attachments/assets/e48b1265-3eda-4a84-9a39-d160b322271a" />

Scaling of data  
let’s scale the Close price data between 0 and 1 using MinMaxScaler to ensure compatibility with the LSTM model
``` python
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])
```
Now, let’s prepare the data for LSTM by creating sequences of a defined length (e.g., 60 days) to predict the next day’s price
```
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(df['Close'].values, seq_length)
```
Train test split of data
``` python
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
```
**LSTM model**
Now, we will build a sequential LSTM model with layers to capture the temporal dependencies in the data
``` python
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
```
Training the lstm model
Now, we will compile the model using an appropriate optimizer and loss function, and fit it into the training data
``` python
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)
```
<img width="724" height="715" alt="image" src="https://github.com/user-attachments/assets/4043cca7-dfe5-4d9c-b824-11941c5b5e35" />

<img width="693" height="238" alt="image" src="https://github.com/user-attachments/assets/d6aea4c9-3b46-4cf0-bd1e-19da842eea7a" />

Now, we need to train the second model. I’ll start by generating lagged features for Linear Regression (e.g., using the past 3 days as predictors)
``` python
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
df = df.dropna()
```
Now, we will split the data accordingly for training and testing
``` python
X_lin = df[['Lag_1', 'Lag_2', 'Lag_3']]
y_lin = df['Close']
X_train_lin, X_test_lin = X_lin[:train_size], X_lin[train_size:]
y_train_lin, y_test_lin = y_lin[:train_size], y_lin[train_size:]
```
**linear regression model**
``` python
lin_model = LinearRegression()
lin_model.fit(X_train_lin, y_train_lin)
```
Now, here’s how to make predictions using LSTM on the test set and inverse transform the scaled predictions
``` python
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
lstm_predictions = lstm_model.predict(X_test_lstm)
lstm_predictions = scaler.inverse_transform(lstm_predictions)
```
Here’s how to generate predictions using Linear Regression and inverse-transform.
``` python
lin_predictions = lin_model.predict(X_test_lin)
lin_predictions = scaler.inverse_transform(lin_predictions.reshape(-1, 1))
```
here’s how to use a weighted average to create hybrid predictions
``` python
min_len = min(len(lstm_predictions), len(lin_predictions))

lstm_predictions_aligned = lstm_predictions[:min_len]
lin_predictions_aligned = lin_predictions[:min_len]

hybrid_predictions = (0.7 * lstm_predictions_aligned) + (0.3 * lin_predictions_aligned)
```
**Predicting using the Hybrid Model**
``` python
lstm_future_predictions = []
last_sequence = X[-1].reshape(1, seq_length, 1)
for _ in range(10):
    lstm_pred = lstm_model.predict(last_sequence)[0, 0]
    lstm_future_predictions.append(lstm_pred)
    lstm_pred_reshaped = np.array([[lstm_pred]]).reshape(1, 1, 1)
    last_sequence = np.append(last_sequence[:, 1:, :], lstm_pred_reshaped, axis=1)
lstm_future_predictions = scaler.inverse_transform(np.array(lstm_future_predictions).reshape(-1, 1))
```
we need to predict the Next 10 Days using Linear Regression
``` python
recent_data = data['Close'].values[-3:]
lin_future_predictions = []
for _ in range(10):
    lin_pred = lin_model.predict(recent_data.reshape(1, -1))[0]
    lin_future_predictions.append(lin_pred)
    recent_data = np.append(recent_data[1:], lin_pred)
lin_future_predictions = scaler.inverse_transform(np.array(lin_future_predictions).reshape(-1, 1))
```
here we need to combine the predictive power of both models to make predictions for the next 10 days
``` python
hybrid_future_predictions = (0.7 * lstm_future_predictions) + (0.3 * lin_future_predictions)
```
Next is to create the final DataFrame to look at the predictions
``` python
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=10)
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'LSTM Predictions': lstm_future_predictions.flatten(),
    'Linear Regression Predictions': lin_future_predictions.flatten(),
    'Hybrid Model Predictions': hybrid_future_predictions.flatten()
})
print(predictions_df)
```
<img width="606" height="271" alt="image" src="https://github.com/user-attachments/assets/18f65e43-f9ad-4e6d-b993-c2f6afb291d7" /><img width="279" height="274" alt="image" src="https://github.com/user-attachments/assets/25138745-deb3-4402-bd94-680634eece41" />


### Insights

- Single-Model Limitations: Traditional linear models often fail to capture sudden market shifts, while deep learning models (like LSTM) can sometimes over-react to short-term volatility.

- Feature Scaling is Critical: Using MinMaxScaler to keep prices between 0 and 1 is essential for the LSTM to process the data without gradient explosion.

- Sequence Importance: A 60-day look-back period provides enough historical context for the model to "remember" relevant price actions.

- Complementary Strengths: LSTM excels at non-linear pattern recognition, while Linear Regression acts as a "sanity check" for the long-term trend.

- Performance Gains: Hybrid models generally show improved stability in performance metrics when the underlying data contains both noise and a clear trend.

### Recommendations

- Implement the Hybrid Approach: Use the combined power of LSTM and Linear Regression when single models show high error rates in validation.

- Standardize Input Sequences: Ensure all time-series data is formatted into consistent sequences (e.g., 60-day windows) before feeding into the hybrid architecture.
- Apply Feature Scaling: Always scale "Close" price data to a $[0, 1]$ range to maintain compatibility with neural network activation functions.
- Monitor Performance Metrics: Regularly compare the Hybrid MSE against the individual model MSEs to ensure the combination is adding value.
- Tune the Sequence Length: Experiment with different window sizes (e.g., 30, 90, or 120 days) to find the optimal historical context for different asset classes.
- Automate the Pipeline: Use Python libraries like pandas, numpy, and scikit-learn to create an automated data preparation pipeline for real-time forecasting.
