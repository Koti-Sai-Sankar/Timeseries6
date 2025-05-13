# Ex.No: 6 HOLT WINTERS METHOD
## Date:
## AIM: 
## ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a HoltWinters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
## PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv('train (1).csv')

data['Order Date'] = pd.to_datetime(data['Order Date'], dayfirst=True, errors='coerce')
data.dropna(subset=['Order Date'], inplace=True)
data.set_index('Order Date', inplace=True)

monthly_sales = data['Sales'].resample('MS').sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot(title="Monthly Sales")
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

decomposition = seasonal_decompose(monthly_sales, model='additive', period=12)
decomposition.plot()
plt.suptitle("Additive Decomposition", fontsize=16)
plt.tight_layout()
plt.show()

train = monthly_sales[:-12]
test = monthly_sales[-12:]

model = ExponentialSmoothing(
train,
trend='add',
seasonal='add',
seasonal_periods=12,
initialization_method='estimated'
)
model_fit = model.fit()

predictions = model_fit.forecast(12)

rmse = sqrt(mean_squared_error(test, predictions))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

final_model = ExponentialSmoothing(
monthly_sales,
trend='add',
seasonal='add',
seasonal_periods=12,
initialization_method='estimated'
).fit()

forecast = final_model.forecast(12)

plt.figure(figsize=(14, 7))
monthly_sales.plot(label='Actual Sales')
forecast.plot(label='Forecast', color='red', linestyle='--')
plt.title('Holt-Winters Forecast - Monthly Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

print(f'Mean Sales: {monthly_sales.mean():.2f}')
print(f'Standard Deviation of Sales: {monthly_sales.std():.2f}')
```
## OUTPUT:
Test Prediction:

![image](https://github.com/user-attachments/assets/1e10f3ee-b14d-4192-981d-249d20fd0cf8)


Final Prediction:

![image](https://github.com/user-attachments/assets/07f8f155-810a-4101-a2e3-99b9c8b5b5b8)

## RESULT:
Thus the program run successfully based on the Holt Winters Method model.
