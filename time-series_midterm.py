#time-series analysis midterm

# Basic packages
import numpy as np 
import pandas as pd
import datetime 

# For Viz
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

# TIME SERIES
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic, kpss, acf
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression

# surpress warnings
import warnings
warnings.filterwarnings("ignore")

sns.set(context='paper', font_scale=1, style='dark')
#%% #to know the encoding of xlsx file
'''
import pandas as pd
import chardet

with open('C:/Users/jimmyhu/Desktop/研究所_行銷/112-2/時間序列分析/onlyCSGOdata_utf8.csv', 'rb') as f:
    result = chardet.detect(f.read())

df = pd.read_csv('C:/Users/jimmyhu/Desktop/研究所_行銷/112-2/時間序列分析/onlyCSGOdata_utf8.csv', encoding=result['encoding'])
'''
#%%
df = pd.read_csv('C:/Users/jimmyhu/Desktop/研究所_行銷/112-2/時間序列分析/onlyCSGOdata_utf8.csv')
# Change the column name into lowercase
df.columns = df.columns.str.lower()
print(df.info())
#%%
#missing value使用Moving Average解決，但程式寫不出來所以用手動計算
#%%
# rename columns
df = df.rename(columns={'month': 'time',
                       'avg. players': 'avg_players',
                       '% gain': 'pctg_gain',
                       'peak players': 'peak_players',
                       'skin price': 'skin_price',
                       'sold units': 'sold_units'})

# change any "Last 30 Days" in the time column into "Oct-21"
df.time = df.time.replace("Last 30 Days", "21-Oct")

# change the time column into date time format
df.time = pd.to_datetime(df.time, format='%y-%b')

# change gain into float
df.gain = pd.to_numeric(df.gain, downcast="float", errors='coerce').map('{:,.2f}'.format)
#%%
# Checking for NAs
print('values missing in the column gain: ',sum(df.gain.isna()), 'and column %gain: ',sum(df.pctg_gain.isna()))
print('amount of missing rows: ', sum(df.isna().any(axis=1)))
print('in total there are ', round((sum(df.isna().any(axis=1)) / len(df)) * 100, 2), '% of rows with missing values')
#%%
df.time.describe()
df.avg_players.describe().apply(lambda x: format(x, 'f'))
#%%

#%%
#plotting time versus average cs player
plt.figure(figsize = (15,8))
sns.lineplot(x = "time", 
           y = "avg_players",
           data = df).set_title("time vs. avg_players")
plt.xlabel("Time")
plt.ylabel("Mean of average number of players")
plt.xlim([datetime.date(2012,9,1), datetime.date(2021,10,1)]);
#%%
#check monthly seasonality
# averaging both game first
df_month = df.groupby('time') \
       .agg({'avg_players':'mean'}) \
       .reset_index() 

# Checking seasonal plot
fig = px.line(x=df_month.time.dt.month, y= df_month.avg_players, 
              color = df_month.time.dt.year, 
              title='Seasonal trend of CS players across months',
              labels=dict(x="Months", y="Average Num of Players"))
fig.update_traces(hovertemplate = "Players: %{y}")
fig.update_layout(showlegend=False)
fig.show()
#this seasonality can't show directly, so we use "jupyter" to solve this question
#%%
ts = df.groupby(["time"])["avg_players"].mean()

def plotMovingAverage(series, window):

    plt.figure(figsize=(16,6))
    plt.plot(series.rolling(window=window,center=False).mean(),label='Rolling Mean', linewidth = 2);
    plt.plot(series.rolling(window=window,center=False).std(),label='Rolling SD', linewidth = 2);
    plt.plot(series[window:], label="Actual values", linewidth = .5);
    plt.title("Moving average \n window size = {}".format(window));
    plt.axhline(y=0, color="black", linestyle='-', linewidth=.5);
    plt.legend();
    
plotMovingAverage(ts, 12) # window being 12 for the 12 months window (i.e annual)
#%%
# Function for Seasonal Trend Decompisition using LOESS (STL)
def plotSTL(data, time, value):
    """value : value from which we want to detrend"""
    
    #Prepare the dataframe
    res_df = data[[time, value]] \
            .groupby([time])[value].mean().reset_index().set_index(time) # Set the index to datetime object
    res_df = res_df.asfreq('MS')
    
    # seasonal trend decomposition
    res_robust = STL(res_df, robust = True).fit();
    res_robust.plot();

plotSTL(df, 'time', 'avg_players')
#%%

#%%
import statsmodels.api as sm
# 將百分比轉換為小數
df['pctg_gain'] = df['pctg_gain'].apply(lambda x: round(float(x.replace('%', '')) / 100, 4))

# 將 gain 中的逗號刪除並轉換為浮點數
for i in range(len(df['gain'])):
    df.loc[i, 'gain'] = float(df.loc[i, 'gain'].replace(',', ''))
df['gain'] = df['gain'].astype(float)

# 使用 toordinal() 將日期轉換為序數
df['time'] = df['time'].apply(lambda x: x.toordinal())

model = LinearRegression(fit_intercept=True)  # 添加截距項

# 訓練模型
model.fit(x, y)

# 打印模型係數
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

x_with_intercept = sm.add_constant(x)
sm_model = sm.OLS(y, x_with_intercept).fit()
print(df.isnull().sum())  # 檢查缺失值
print(df.isna().sum())    # 檢查無效值
# 獨立變量（特徵）
x = df[['time', 'gain', 'pctg_gain', 'peak_players', 'skin_price', 'sold_units']].values

# 依賴變量（目標）
y = df['avg_players'].values

# 創建線性回歸模型
model = LinearRegression()
model.fit(x, y)

# 添加截距項
x = sm.add_constant(x)

# 適配線性回歸模型
model = sm.OLS(y, x).fit()

# 印出回歸模型的摘要
print(model.summary())
#%%
# Function for Seasonal Trend Decompisition using LOESS (STL)利用LOESS(局部加權迴歸)方法對時間序列數據使用季節性趨勢分解
def plotRegression(data, time, value):
    """value : value from which we want to detrend"""
    
    # Prepare the dataframe
    res_df = data[[time, value]] \
            .groupby([time])[value].mean().reset_index().set_index(time) # Set the index to datetime object
    res_df = res_df.asfreq('MS')
    
    # Regression analysis
    X = np.arange(len(res_df)).reshape(-1, 1)  # Independent variable (time)
    y = res_df[value].values.reshape(-1, 1)    # Dependent variable (avg_players)
    
    # Fit linear regression model
    X = sm.add_constant(X)  # Add constant term for intercept
    model = sm.OLS(y, X).fit()
    
    # Print regression summary
    print(model.summary())
    
    # Plot original data
    plt.figure(figsize=(10, 6))
    plt.scatter(res_df.index, res_df[value], label='Original Data', color='blue')
    
    # Plot regression line
    predictions = model.predict(X)
    plt.plot(res_df.index, predictions, label='Regression Line', linestyle='--', color='red')
    
    plt.title('Regression Analysis')
    plt.xlabel('Time')
    plt.ylabel(value)
    plt.legend()
    plt.grid(True)
    plt.show()

plotRegression(df, 'time', 'avg_players')
#%%
import statsmodels.api as sm
df['pctg_gain'] = df['pctg_gain'].apply(lambda x: round(float(x.replace('%', '')) / 100, 4))
for i in range(len(df['gain'])):
    df.loc[i, 'gain'] = float(df.loc[i, 'gain'].replace(',', ''))
    
# Independent variables (features)
df['time_ordinal'] = df['time'].apply(lambda x: x.toordinal())

# Define independent variables (features)
X = df[['time_ordinal', 'gain', 'pctg_gain', 'peak_players', 'skin_price', 'sold_units']].values

# Dependent variable (target)
y = df['avg_players'].values

# Fit linear regression model
model = sm.OLS(y, x).fit()
model.fit(X, y)

# Print model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print(model.summary())
#%%
#each var's descriptive
plt.figure(figsize = (15,8))
sns.lineplot(x = "time", 
           y = "avg_players",
           data = df).set_title("General Trend: Top Steam Game's Average Num of Players")
plt.xlabel("Time");
plt.ylabel("Mean of average number of players");
plt.xlim([datetime.date(2013,1,1), datetime.date(2021,10,1)]);

plt.figure(figsize=(15, 10))
for column in df.columns:
    plt.subplot(3, 3, list(df.columns).index(column) + 1)
    sns.histplot(df[column], kde=True)
    plt.title(column)
plt.tight_layout()

plt.xlim([datetime.date(2013,1,1), datetime.date(2021,10,1)]);
plt.show()

plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns):
    plt.subplot(3, 3, i + 1)
    sns.lineplot(x='time', y=column, data=df)
    plt.title(column)
    plt.xlim([datetime.date(2012, 9, 1), datetime.date(2021, 10, 1)])
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# Scatter plot for gain
plt.subplot(1, 2, 1)
plt.scatter(df['time'], df['gain'])
plt.title('Scatter Plot of Gain over Time')
plt.xlabel('Time')
plt.ylabel('Gain')

# Scatter plot for pctg_gain
plt.subplot(1, 2, 2)
plt.scatter(df['time'], df['pctg_gain'])
plt.title('Scatter Plot of Percentage Gain over Time')
plt.xlabel('Time')
plt.ylabel('Percentage Gain')

plt.tight_layout()
plt.show()

