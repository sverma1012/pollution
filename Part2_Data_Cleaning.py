#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn


# In[2]:


pollution = pd.read_excel('PM2.5climate.xlsx')
pollution.head()


# In[3]:


pollution.shape


# Because the data have been preprocess, most of the variables will not be changed.
# For data cleaning, we will check duplicated and empty variables and then make a decision whether to 
# keep, change or drop the variables

# In[4]:


print('Total Missing values:', round((pollution['pm2.5'].isnull().sum()/43824)*100,2),'%')


# As seen in the exploratory data, only one attribute have 2061 missing values, which is 0.047 or 4.7% of the total dataset observations.

# ---

# ## Dropping Variables
# 
# ## Missing Values

# In[5]:


pollution.drop(columns = ['No','hour', 'year', 'day'], inplace=True)
pollution.head()


# In[6]:


pollution.dropna(inplace = True)
pollution.reset_index(drop = True, inplace = True)
pollution.head()


# In[7]:


pollution['pm2.5'].isnull().sum()


# In[8]:


pollution.shape


# All the missing values have been dropped because, as observed in the exploration, the missing values are in the target variable. If they had not been dropped, that would skew our regression results because it would not give any prediction for the level of pollution.
# 
# In addition, the missing values will not overall impact the dataset as it accounts for only 5% total. We chose not to replace the missing values because it was such a small portion of the dataset and replacing it could cause bias and it was a safer option to remove them than replace them.
# 
# No. has been dropped because it does not give any information since it is simply the row number of the observation.
# 
# Hour was dropped because it is a minute detail in pollution levels and we want to explore larger trends in the level s of pollution. We want to understand the long-run trend in the level of pollution and hour is a very short-term measure.
# 
# Similarly, day is also a short-term measure that we are not interested in exploring. 
# 
# Year was dropped because in our exploration it was found that it was not correlated with the level fo pollution and, hence, we decided to drop it.

# Let us rename the variables for better understanding:

# In[9]:


pollution.columns


# In[10]:


names = dict(DEWP = 'dewTemp', TEMP = 'temp', PRES = 'pres', cbwd = 'windDir',
             Iws = 'windSpeed', Is = 'cumSnow', Ir = 'cumRain')
pollution.rename(columns = names, inplace = True)
pollution.columns


# In[11]:


pollution['windDir'].replace({'cv':'SW'}, inplace = True)
pollution['windDir'].value_counts()


# #### About the project
# 
# The next step will be exploring the outliers in the dataset.
# First part of this visualization will be shown with outliers included and the second part the visualizations will be plotted after removing outleirs. 
# 
# For the data analysis, there will be two regression models shown. The first model will be the model with outliers and the second model is after removing outliers. Towards the end, we will be comparing both models and see which one is most reliable model to be used for the predictions
# 

# ---

# ## Outliers

# In[12]:


from scipy import stats
import numpy as np

z2 = np.abs(stats.zscore(pollution[['month', 'pm2.5', 'dewTemp', 'temp', 'pres', 'windSpeed', 'cumSnow', 'cumRain']]))
print(z2)


# In[13]:


threshold = 3
print(np.where(z2 > 3))


# In[14]:


outliers2 = pollution[(z2 >=3)]
outliers2


# In[15]:


pd.set_option('display.max_rows', 1000)
other2 = pollution[(z2 < 3).all(axis=1)]
other2


# The first table shows the values of the outliers and the second table shows the values of the dataset without the outliers. Looking at these values, we can see that the values of cumSnow and cumRain are the same for both tables (outliers and non-outliers). We do not see a reason to find z-scores of these two columns since their values are close and removing them based on their z-scores would be unreasonable causing some bias. 
# 
# Hence, let us find outliers based on the z-scores of all numerical features except cumSnow and cumRain.

# In[16]:


from scipy import stats
import numpy as np

z = np.abs(stats.zscore(pollution[['month', 'pm2.5', 'dewTemp', 'temp', 'pres', 'windSpeed']]))
print(z)


# In[17]:


threshold = 3
print(np.where(z > 3))


# In[18]:


outliers = pollution[(z >=3)]
outliers.shape


# In[19]:


pd.set_option('display.max_rows', 1000)
other = pollution[(z < 3).all(axis=1)]
other


# Let us create a new dataframe without the outliers:

# In[20]:


pollution.shape


# In[21]:


pollution2 = pollution[(z < 3).all(axis=1)]
pollution2.shape


# In[22]:


pollution2.reset_index(drop = True, inplace = True)
pollution2.head(10)


# Removing outliers based on numerical features, except cumSnow and cumRain, results in fewer outliers being removed; approximately 2,000 rows have not been removed compared to the rows removed when cumSnow and cumRain were used in z-score calculations. This is interesting because it shows that values that are not much different from other observations can be removed due to minor differences. Hence, it is important to look at the outliers to determine if they should be removed for further model creation.

# In[23]:


print('The data frame without outliers is', round((pollution2.shape[0]/pollution.shape[0]),2),      '% smaller than the original dataframe')


# ---

# ## One-Hot Encoding

# Let us one-hot encode the windDir variable since it is categorical. 
# 
# We have to one-hot encode this variable because it has string values and we will give 0, 1 values to each value of the variable.

# ##### Dataset with Outliers

# In[24]:


from sklearn.preprocessing import OneHotEncoder

windDirection = OneHotEncoder()
windDirectionDF = windDirection.fit_transform(pollution[['windDir']])
windDirectionDF = pd.DataFrame(windDirectionDF.toarray())
windDirectionDF.columns = windDirection.get_feature_names()
windDirectionDF.head()


# In[25]:


pollution.drop(columns = ['windDir'], inplace = True)
pollution.head()


# In[26]:


pollutionDF = pd.concat([pollution, windDirectionDF], axis = 1)
pollutionDF.head()


# In[27]:


pollutionDF.columns


# In[28]:


colnames = ['month', 'dewTemp', 'temp', 'pres', 'windSpeed', 'cumSnow', 'cumRain', 'x0_NE', 'x0_NW', 'x0_SE', 'x0_SW', 'pm2.5']
pollutionDF = pollutionDF[colnames]
pollutionDF.head()


# ---

# ##### Dataset without Outliers

# In[29]:


windDirection_wo = OneHotEncoder()
windDirection_woDF = windDirection_wo.fit_transform(pollution2[['windDir']])
windDirection_woDF = pd.DataFrame(windDirection_woDF.toarray())
windDirection_woDF.columns = windDirection_wo.get_feature_names()
windDirection_woDF.head()


# In[30]:


pollution2.drop(columns = ['windDir'], inplace = True)
pollution2.head()


# In[31]:


pollution2DF = pd.concat([pollution2, windDirection_woDF], axis = 1)
pollution2DF.head()


# In[32]:


colnames = ['month', 'dewTemp', 'temp', 'pres', 'windSpeed', 'cumSnow', 'cumRain', 'x0_NE', 'x0_NW', 'x0_SE', 'x0_SW', 'pm2.5']
pollution2DF = pollution2DF[colnames]
pollution2DF.head()


# In[33]:


pollution2DF.shape


# In[34]:


pollutionDF.shape


# After one-hot encoding the variables for the datasets with and without outliers, it can be seen that the number of observations in both the dataframes have remained the same. However, the number of columns have increased because of the one-hot encoded columns.

# ---

# ## Visualizations
# ##### Dataset with Outliers

# #### Scatterplot
# Let us explore if there are any outliers in this dataset with scatterplot:

# In[37]:


fig, ax = plt.subplots(4, 2, figsize = (10, 20))
#fig.tight_layout()

ax[0,0].scatter(y = pollution['pm2.5'], x = pollution['month'])
ax[0,0].set_xlabel('Month')
ax[0,0].set_ylabel('PM2.5')
ax[0,0].set_title('Month v/s PM 2.5 levels')

ax[0, 1].scatter(y = pollution['pm2.5'], x = pollution['dewTemp'])
ax[0, 1].set_xlabel('Dew Point Temperature')
ax[0, 1].set_ylabel('PM2.5')
ax[0, 1].set_title('Dew Point v/s PM 2.5 levels')

ax[1,0].scatter(y = pollution['pm2.5'], x = pollution['temp'])
ax[1,0].set_xlabel('Temperture')
ax[1,0].set_ylabel('PM2.5')
ax[1,0].set_title('Temperature v/s PM 2.5 levels')

ax[1,1].scatter(y = pollution['pm2.5'], x = pollution['pres'])
ax[1,1].set_xlabel('Air Pressure')
ax[1,1].set_ylabel('PM2.5')
ax[1,1].set_title('Air Pressure v/s PM 2.5 levels')

ax[2,0].scatter(y = pollution['pm2.5'], x = pollution['windSpeed'])
ax[2,0].set_xlabel('Wind Speed')
ax[2,0].set_ylabel('PM2.5')
ax[2,0].set_title('Wind Speed v/s PM 2.5 levels')

ax[2,1].scatter(y = pollution['pm2.5'], x = pollution['cumSnow'])
ax[2,1].set_xlabel('Hours of Cumulated Snow')
ax[2,1].set_ylabel('PM2.5')
ax[2,1].set_title('Hours of Cumulated Snow v/s PM 2.5 levels')

ax[3,0].scatter(y = pollution['pm2.5'], x = pollution['cumRain'])
ax[3,0].set_xlabel('Hours of Cumulated Rain')
ax[3,0].set_ylabel('PM2.5')
ax[3,0].set_title('Hours of Cumulated Rain v/s PM 2.5 levels')

plt.show()


# Looking at the above dashboard, it is visible that there are a few outliers in the month, dew point, temperature, and air pressure variables. This is because there are a few points that have much higher pm2.5 levels for certain x-values. This shows that there are certain observations where there are exceptionally high levels of air pollution for a certain month in a year. Similarly, there are certain temperature, dew point temperature, and air pressure values for which there are exceptionally high air pollution levels. These high values of air pollution are probably influenced by external factors that were not controlled in this dataset. Hence, it is probably better to remove any outliers. However, let us see how many outliers there are in totality.
# 
# The scatterplots of the wind speed, cumulated hours fo snow and rain are interesting because they follow a 'L' pattern because of which it is hard to distinguish if there are outliers. However, outliers in these variables will be able to be seen with the z-value calculations.

# #### Box plot

# In[38]:


fig, ax = plt.subplots(4, 2, figsize = (10, 20))
ax[0,0].boxplot(x=pollution['month'])
ax[0,0].set_title('Month')

ax[0,1].boxplot(x=pollution['pm2.5'])
ax[0,1].set_title('PM2.5 Level')

ax[1,0].boxplot(x=pollution['dewTemp'])
ax[1,0].set_title('Dew Point Temperature')

ax[1,1].boxplot(x=pollution['temp'])
ax[1,1].set_title('Temperature')

ax[2,0].boxplot(x=pollution['pres'])
ax[2,0].set_title('Air Pressure') 

ax[2,1].boxplot(x=pollution['windSpeed'])
ax[2,1].set_title('Wind Speed') 

ax[3,0].boxplot(x=pollution['cumSnow'])
ax[3,0].set_title('Hours of Cumulative Snow') 

ax[3,1].boxplot(x=pollution['cumRain'])
ax[3,1].set_title('Hours of Cumulative Rain') 

plt.show()


# The boxplot above are before the outliers were removed. 
# 
# There are four variables that does not seem to show outliers, these variables are month, dew point temperature, temperature and air pressure. Although they seem to not have outliers, they do have some skewness which might indicate some higher-than-normal values.

# ##### Dataset without outliers
# #### Scatterplot

# In[39]:


fig, ax = plt.subplots(4, 2, figsize = (10, 20))
#fig.tight_layout()

ax[0,0].scatter(y = pollution2['pm2.5'], x = pollution2['month'])
ax[0,0].set_xlabel('Month')
ax[0,0].set_ylabel('PM2.5')
ax[0,0].set_title('Month v/s PM 2.5 levels')

ax[0, 1].scatter(y = pollution2['pm2.5'], x = pollution2['dewTemp'])
ax[0, 1].set_xlabel('Dew Point Temperature')
ax[0, 1].set_ylabel('PM2.5')
ax[0, 1].set_title('Dew Point v/s PM 2.5 levels')

ax[1,0].scatter(y = pollution2['pm2.5'], x = pollution2['temp'])
ax[1,0].set_xlabel('Temperture')
ax[1,0].set_ylabel('PM2.5')
ax[1,0].set_title('Temperature v/s PM 2.5 levels')

ax[1,1].scatter(y = pollution2['pm2.5'], x = pollution2['pres'])
ax[1,1].set_xlabel('Air Pressure')
ax[1,1].set_ylabel('PM2.5')
ax[1,1].set_title('Air Pressure v/s PM 2.5 levels')

ax[2,1].scatter(y = pollution2['pm2.5'], x = pollution2['windSpeed'])
ax[2,1].set_xlabel('Wind Speed')
ax[2,1].set_ylabel('PM2.5')
ax[2,1].set_title('Wind Speed v/s PM 2.5 levels')

ax[3,0].scatter(y = pollution2['pm2.5'], x = pollution2['cumSnow'])
ax[3,0].set_xlabel('Hours of Cumulated Snow')
ax[3,0].set_ylabel('PM2.5')
ax[3,0].set_title('Hours of Cumulated Snow v/s PM 2.5 levels')

ax[3,1].scatter(y = pollution2['pm2.5'], x = pollution2['cumRain'])
ax[3,1].set_xlabel('Hours of Cumulated Rain')
ax[3,1].set_ylabel('PM2.5')
ax[3,1].set_title('Hours of Cumulated Rain v/s PM 2.5 levels')

plt.show()


# The scatterplot above is visualizing the dataset after the outliers are removed. Due to the large dataset, the plot in the dataset seem to be cluster and does not appears to show linear correlation between variables
# 
# The graph between Temperature v/s PM2.5 level and the Air Pressure v/s PM2.5 level appears to be very interesting as they seem to follow the same pattern. Both are clustered in the middle. While the Wind Speed v/s PM 2.5 Level appears to cluster more to the left and the Dew Point v/s PM2.5 Level are the opposite, it clusters more to the left
# 

# #### Box plot

# In[41]:


fig, ax = plt.subplots(4, 2, figsize = (10, 20))
ax[0,0].boxplot(x=pollution2['month'])
ax[0,0].set_title('Month')

ax[0,1].boxplot(x=pollution2['pm2.5'])
ax[0,1].set_title('PM2.5 Level')

ax[1,0].boxplot(x=pollution2['dewTemp'])
ax[1,0].set_title('Dew Point Temperature')

ax[1,1].boxplot(x=pollution2['temp'])
ax[1,1].set_title('Temperature')

ax[2,0].boxplot(x=pollution2['pres'])
ax[2,0].set_title('Air Pressure') 

ax[2,1].boxplot(x=pollution2['windSpeed'])
ax[2,1].set_title('Wind Speed') 

ax[3,0].boxplot(x=pollution2['cumSnow'])
ax[3,0].set_title('Hours of Cummulative Snow') 

ax[3,1].boxplot(x=pollution2['cumRain'])
ax[3,1].set_title('Hours of Cummulative Rain') 

plt.show()


# The boxplot above are after the ouliers removed. Similar to the previous boxplot with outlier the variables month, dew point temperature, temperature and air pressure still does not show any outliers. 
# 
# There are some variables that shows some adjustments or changes without the outliers. Both Hours of Cummulative Snow and Hours of Cummulative rain seems to show less points. These changes are also seen in PM2.5 level and Wind Speed

# ---

# ## Scaling Numerical Features

# ##### Dataset with Outliers

# In[42]:


pollutionDF.columns


# In[43]:


from sklearn.preprocessing import StandardScaler

cols = ['month', 'dewTemp', 'temp', 'pres', 'windSpeed', 'cumSnow', 'cumRain']
scaler = StandardScaler()
pollutionDF[cols] = scaler.fit_transform(pollutionDF[cols])
pollutionDF.head()


# ---

# ##### Dataset without outliers

# In[44]:


pollution2DF.columns


# In[45]:


cols = ['month', 'dewTemp', 'temp', 'pres', 'windSpeed', 'cumSnow', 'cumRain']
scaler = StandardScaler()
pollution2DF[cols] = scaler.fit_transform(pollution2DF[cols])
pollution2DF.head()


# We standardized the dataset with and without outliers. This is because we wanted to bring all the numerical values to the same scale. We standardized it instead of normalize the datasets because it would let us know how many standard deviations the observations are from the mean. 
# 
# We chose not to standardize the dependent variable for ease of interpretation.

# ---

# We have completed the process of cleaning our data and preparing it for the regression analysis and visualizations (storytelling).

# In[46]:


pollutionDF.to_excel('pollution_outliers.xlsx')
pollution2DF.to_excel('pollution_NoOutliers.xlsx')


# ---

# ### Let us now move on to Data Analysis
