#!/usr/bin/env python
# coding: utf-8

# ### TMU ITM618 Fall 2022

# Your task is to develop a series of research hypotheses to predict recession based on theory or past empirical evidence and then apply some of the techniques covered in class to such data for testing.
# 
# Smoothed recession probabilities for the United States are obtained from a dynamic-factor Markov-switching model applied to four monthly coincident variables: non-farm payroll employment, the index of industrial production, real personal income excluding transfer payments, and real manufacturing and trade sales.
# 
# You can use the 10 Year-3 Month Treasury Yield Spread, which is the difference between the ten-year treasury rate and the three-month treasury rate. This spread is widely used as a gauge to study the yield curve. A 10-year-3-month treasury spread that approaches 0 signifies a "flattening" yield curve. Furthermore, a negative 10-year-3-month spread has historically been viewed as a precursor or predictor of a recessionary period. The New York Fed uses the rate in a model to predict recessions 2 to 6 quarters ahead.

# ### Data Set

# Students are encouraged to collect data by themselves. Students can collect data from any sources
# such as the following, but not limit to:
# * TMU Library
# * Yahoo Finance
# * Or any other source

# ### Analysis and Report

# Students should work closely in a group on data collection, data analysis and result interpretation, report writing, etc. In the project report, students are supposed to describe the results and conclusion of their analysis. Keep in mind that plots, tables and other visual representations of data are useful in conveying your conclusions. In addition, you may want to include the following parts in your reports.
# 
# Questions/Hypotheses: Write one or multiple questions or hypotheses you want to explore with the data sets. After each question, state your expected answers, which may be different from your data analysis because you have not yet analyzed the data.
# 
# Data Description: Describe the data sets. What is the data, e.g., variables and results? How was the data collected? Briefly summarize the data. Provide the URL link if available.
# 
# Methodologies: Write a complete, clear description of the analysis you performed. This should be sufficient for someone else to write an R program to reproduce your results. It should also likely be helpful to people who read your code later. This section should tie your computations to your questions/hypotheses, indicating exactly what results would lead you to what conclusion. You may want to provide the key statistics, e.g., t-statistic, z-statistic, p-values, R2 and the adjusted R2, etc.
# 
# Results and Conclusion: Discuss your results. Focus in particular on the results that are most interesting, surprising, or important. Discuss the consequences or implications. Interpret the results: if the answers are unexpected, then see whether you can find an explanation for them, such as an external factor that your analysis did not account for. You may also want to make prediction for new scenarios.
# 
# Appendix: Put plots, tables, technical details or other results in appendix if necessary.

# ### Presentation

# Each group should select one or multiple team members to present their projects in class. Each presentation should be no longer than 10 minutes. It is encouraged to use slides. The slide deck should summarize the main points of your project, including motivation, research questions, and results using, using Guy Kawasaki's presentation framework.
# 
# During the presentations, all students from not-presenting groups should actively ask questions. Each member of the presenting group, not only the presenters, can answer the questions or give comments.

# ### Project Submission

# The group will submit once via D2L. Your submission should include a Jupyter notebook that includes a project report, Python script, and another zip file that includes the data set and the slides. It is suggested that the project reports should be 500-word maximum, excluding the appendix.

# In[1]:


pip install stats-can


# In[2]:


pip install yfinance==0.1.86


# In[3]:


from datetime import datetime

import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import scipy as sp
import scipy.stats as stats
import stats_can
import statsmodels.api as sm

import yfinance as yf


# In[4]:


SP= yf.Ticker("VOO")
SP

#leading up to, and during a recession, prices of stocks tend to decline ... add on to this 
#for this reason, we chose to use the S&P500 to represent an overview of the stock market. 
#Since 1950, the average decline for the S&P 500 during a recession is about 29%, 
#he said. So far this year, the S&P 500 has fallen nearly 24%.


# In[5]:


View = SP.history(start="2022-01-02", end="2022-10-07")
View


# In[6]:


data = yf.download("VOO VNQ NQ=F", start="2022-01-02", end="2022-10-07")
data


#Industries affected most include retail, restaurants, travel/tourism, 
#leisure/hospitality, service purveyors, real estate, & manufacturing/warehouse.

#VOO is the S&P500, VNQ is a Real estate ETF, NQ=F is also similar to the S&P 500 but it combines 100 of the top preforming internation stocks. 


# In[7]:


data['Adj Close','NQ=F'].plot(figsize =[15, 7.5])
plt.title('ADJ Close for NASDAQ 100 (NQ=F)')


# In[8]:


data['Adj Close','VOO'].plot(figsize=[15, 7.5])
data['Adj Close','VNQ'].plot(figsize=[15, 7.5])

plt.title('ADJ Close for Vanguard 500 (VOO) and Vanguard Real Estate Index 1 (VNQ)')


# In[9]:


eco_vec_map = {'GDP':'v65201210',
               'Unemployment_Rate':'v2440389'}


# In[10]:


#periods can be changed to adjust time period being plotted
vectors = list(eco_vec_map.values())
df = stats_can.sc.vectors_to_df(vectors, periods = 300)


# In[11]:


inv_map = {v: k for k, v in eco_vec_map.items()}
df.columns = df.columns.to_series().map(inv_map)
df.index.names = ['Date']


# In[12]:


df.plot(subplots = True, figsize = (14,8), layout = (2,1))


# In[23]:


df = pd.read_csv("Rates_data.csv")


# In[24]:


x = []
y = []


# In[25]:


fig = plt.figure(figsize= [12, 5])
df['V122667775'].plot()
plt.title('Interest Rates')
plt.show()


# In[ ]:





# Smoothed recession probabilities for the United States are 
# obtained from a dynamic-factor Markov-switching model applied to four
# monthly coincident variables: non-farm payroll employment, the index
# of industrial production, real personal income excluding transfer
# payments, and real manufacturing and trade sales.

# In[26]:


#Datasets

df_e2 = pd.read_csv("Non-Farm Payroll Employment_%.csv", header=0, index_col=0, infer_datetime_format=True, parse_dates=['DATE'], dayfirst=True)
df_index = pd.read_csv("The Index of Industrial Production.csv", header=0, index_col=0, infer_datetime_format=True, parse_dates=['DATE'], dayfirst=True)
df_income = pd.read_csv("Real Personal Income Excluding Current Transfer Receipts.csv", header=0, index_col=0, infer_datetime_format=True, parse_dates=['DATE'], dayfirst=True)
df_sales = pd.read_csv("Real Manufacturing and Trade Sales.csv", header=0, index_col=0, infer_datetime_format=True, parse_dates=['DATE'], dayfirst=True)
df_rec = pd.read_csv("JHDUSRGDPBR.csv", header=0, index_col=0, infer_datetime_format=True, parse_dates=['DATE'], dayfirst=True)


# In[17]:


df_e2


# In[18]:


df_index


# In[19]:


df_income


# In[22]:


df_sales


# In[27]:


df_rec


# In[28]:


#The Graphs for % Change

fig = plt.figure(figsize= [12, 5])
df_e2['% chg'].plot()
plt.title('% CHG of Non-Farm Payroll Employment')
plt.show()
 
fig = plt.figure(figsize= [12, 5])
df_index['% chg'].plot()
plt.title('% CHG of The Index of Industrial Production')
plt.show()

fig = plt.figure(figsize= [12, 5])
df_income['% chg'].plot()
plt.title('% CHG of Real Personal Income Excluding Current Transfer Receipts')
plt.show()
 
fig = plt.figure(figsize= [12, 5])
df_sales['% chg'].plot()
plt.title('% CHG of Real Manufacturing and Trade Sales')
plt.show()


# In[29]:


#Fitting the Markov Model

msdr_model = sm.tsa.MarkovRegression(endog=df_index['% chg'], k_regimes=2, trend='c', exog=df_sales['% chg'], switching_variance=True)
msdr_model_results = msdr_model.fit(iter=10000)


# In[30]:


#Markov Model of Non-Farm Payroll Employment

figure, axes = plt.subplots(3, figsize=(20, 10))

ax = axes[0]
ax.plot(df_e2.index, df_e2['% chg'])
ax.set(title='% CHG of Non-Farm Payroll Employment')

ax = axes[1]
ax.plot(df_e2.index, msdr_model_results.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of regime 0')
 
ax = axes[2]
ax.plot(df_e2.index, msdr_model_results.smoothed_marginal_probabilities[1])
ax.plot(df_rec.index, df_rec['JHDUSRGDPBR'])
ax.set(title='Smoothed probability of regime 1 super-imposed on Recession Indicator  (Orange)')
 
plt.show()


# In[31]:


#Markov Model of The Index of Industrial Production

figure, axes = plt.subplots(3, figsize=(20, 10))

ax = axes[0]
ax.plot(df_index.index, df_index['% chg'])
ax.set(title='% CHG of The Index of Industrial Production')

ax = axes[1]
ax.plot(df_index.index, msdr_model_results.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of regime 0')
 
ax = axes[2]
ax.plot(df_index.index, msdr_model_results.smoothed_marginal_probabilities[1])
ax.plot(df_rec.index, df_rec['JHDUSRGDPBR'])
ax.set(title='Smoothed probability of regime 1 super-imposed on Recession Indicator (Orange)')
 
plt.show()


# In[32]:


#Markov Model of Real Personal Income Excluding Current Transfer Receipts
figure, axes = plt.subplots(3, figsize=(20, 10))

ax = axes[0]
ax.plot(df_income.index, df_income['% chg'])
ax.set(title='% CHG of Real Personal Income Excluding Current Transfer Receipts')

ax = axes[1]
ax.plot(df_income.index, msdr_model_results.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of regime 0')
 
ax = axes[2]
ax.plot(df_income.index, msdr_model_results.smoothed_marginal_probabilities[1])
ax.plot(df_rec.index, df_rec['JHDUSRGDPBR'])
ax.set(title='Smoothed probability of regime 1 super-imposed on Recession Indicator (Orange)')
 
plt.show()


# In[33]:


#Markov Model of Real Manufacturing and Trade Sales

figure, axes = plt.subplots(3, figsize=(20, 10))

ax = axes[0]
ax.plot(df_sales.index, df_sales['% chg'])
ax.set(title='% CHG of Real Manufacturing and Trade Sales')

ax = axes[1]
ax.plot(df_sales.index, msdr_model_results.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of regime 0')
 
ax = axes[2]
ax.plot(df_sales.index, msdr_model_results.smoothed_marginal_probabilities[1])
ax.plot(df_rec.index, df_rec['JHDUSRGDPBR'])
ax.set(title='Smoothed probability of regime 1 super-imposed on Recession Indicator (Orange)')
 
plt.show()

