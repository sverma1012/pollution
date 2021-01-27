#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# ## By Jorania F. Alves and Sneha Verma

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pollution = pd.read_excel('PM2.5climate.xlsx')
pollution.head()


# The top 5 rows shows that the target variable (pm2.5) has NaN (missing) values. This helps us understand that we need to look for missing values in our dataset, especially the target variable.

# ---

# ## Project Description

# The data describes the pollution level depending on various air quality features. There are 43,824 observations with 13 columns (including the target variable). We intend to carry out regression on this dataset to predict the pollution level (PM2.5 level) through various air quality features.
# 
# The features are as follows:
# 1) No: row number (Quantitative)
# 
# 2) Year: year of data (Quantitative)
# 
# 3) Month: month of data (Quantitative)
# 
# 4) Day: day of the data (Quantitative)
# 
# 5) Hour: hour of data (Quantitative)
# 
# 6) PM2.5: PM2.5 concentration in micrograms per cubic meter of air (Quantitative)
# 
# 7) DEWP: dew point; atmospheric temperature below which water droplets begin to condense varies based on pressure and humidity in celsius(Quantitative)
# 
# 8) TEMP: temperature in celsius(Quantitative)
# 
# 9) PRES: air pressure in pascals (Quantitative)
# 
# 10) cbwd: combined wind direction (Quantitative)
# 
# 11) lws: cumulated wind speed in meters/second(Quantitative)
# 
# 12) ls: cumulated hours of snow (Quantitative)
# 
# 13) lr: cumulated hours of rain (Quantitative)
# 
# The dataset is already preprocessed to some degree (it is not raw data). According to the description, there are a few missing values in the target variable (which will be dealt with in the project cleaning).

# ---

# ## Basic Exploration of Dataset

# In[3]:


pollution


# In[4]:


pollution.shape


# In[5]:


pollution.columns


# There are 43,824 rows with 13 features, as described above. This is a large dataset.

# In[6]:


pollution.describe()


# All the variables are numerical and thus can be described.
# 
# The column 'No' does not have any real implications since it gives the row number (which can be provided by the index), hence it will probably be removed.
# 
# The year column is interesting because it gives the "mean" year which might not help in this context. We might decide to categorize the variable or to create dummy variables or simply remove the variable.
# 
# Similarly, the month, day and hour variables are interesting and we may use the month and day variables because it will allow us to understand the months and days in which there are high levels of pollution. However, the hour variable is extrmely detailed and may end up causing confusion or bias.
# 
# pm2.5 is our target variable that describes the levels of fine particulate matter in the air which is hazardous for people's health in micrograms per cubic meter of air. The mean level in Beijing is 98.6 micrograms per cubic meter and the median is 72 micrograms per cubic unit. Looking at this large difference we can assume that there might be outliers in this variable.
# 
# DEWP and TEMP are in degrees celsius. DEWP has a mean value of 1.82 degrees celsius and a median of 2 degrees. It does not seem like there are many or any outliers. TEMP has a mean of 12.45 degrees with a median of 14 degrees. While there are some differences between teh mean and median, there is not a large difference indicating few outliers.
# 
# PRES is the air pressure in pascals. It has a mean of 1016.45 pascals and a median of 1016 pascals. This is approximately the same values (no indication of outliers).
# 
# lws is the cumulated wind speed in meters per second. The mean and median are close to each other with a difference of 2 units. 
# 
# Similarly, ls and lr record the cumulated hours of snow and rain, respectively. The values of the mean and median for both of these variables are close to each other indicating that there are no outliers (or close to none).

# Let us now look at the unique values of the categorical variable:

# In[7]:


pollution['cbwd'].value_counts()


# There is a value called 'cv' which, upon reserach, has been found to be equivalent to SW (Southwest). Hence, we will change that once we start cleaning the data.

# Let us now look at the types of the data points:

# In[8]:


pollution.info()


# Through this table, we can see that most variables are integars with three variables as floating numbers (pm2.5, TEMP, and PRES) and one categorical variable of type 'object' (cbwd). 
# 
# It can also be seen that the target variable has missing values which will need to be dealt in the data cleaning process.

# ---

# ## Missing Values

# In[9]:


pollution.isnull().sum()


# Only one variable, pm2.5, has missing values. There are approximately 2000 missing rows.
# 
# Let us look at visualizations of the features to understand if there are any outliers and to see the distribution follows any pattern.

# In[10]:


pollution.duplicated().value_counts()


# It appears that the dataset does not have any duplicated values.

# ---

# ## Distribution of Features

# ### year

# In[11]:


pollution['year'].value_counts()


# In[12]:


pollution['year'].value_counts().plot(kind = 'bar')
plt.title('Distribution of Years')
plt.xlabel('year')
plt.xticks(rotation = 0)
plt.ylabel('Count')
plt.show()


# Looking at this plot and the value counts table, one can see that there are nearly similar amount of observations for each year. This shows a uniform distribution and tells us that the observations are equitable according to year.

# ### month

# In[13]:


pollution['month'].value_counts()


# In[14]:


pollution['month'].value_counts().plot(kind = 'bar')
plt.title('Distribution of Month')
plt.xlabel('month')
plt.xticks(rotation = 0)
plt.ylabel('Count')
plt.show()


# This distribution is similar to the distribution to the 'year' variable. The distribution is similar (or close) to the distribution of month. All months (across all years) have the same number of observations.
# 
# This is good bcecause we do not need to upscale the data to account for missing observations for the month.

# In[15]:


tab1 = pollution.groupby(['year', 'month'])
tab1['month'].value_counts()


# This table shows that for every year's every month, there are similar amounts of rows. 

# ### Day

# In[16]:


pollution['day'].value_counts()


# In[17]:


plt.figure(figsize = (8,5))
pollution['day'].value_counts().plot(kind = 'bar')
plt.title('Distribution of Days')
plt.xlabel('day')
plt.xticks(rotation = 0)
plt.ylabel('Count')
plt.show()


# The table and graph shows that most days also have similar amounts of observations. However, the 31st day of a month have the least amount of observations over all five years.

# ### hour

# In[18]:


pollution['hour'].value_counts()


# In[19]:


pollution['hour'].value_counts().plot(kind = 'bar')
plt.title('Distribution of Days')
plt.xlabel('day')
plt.xticks(rotation = 0)
plt.ylabel('Count')
plt.show()


# The table and graph shows that there are similar amounts of observations for each day of every year of every month. 

# ### pm2.5

# In[20]:


# Understand the maximum and minimum values of the feature to understand possible values of bins
print(np.nanmax(pollution['pm2.5']))
print(np.nanmin(pollution['pm2.5']))

# np.nan is used because there are NaN values in this column.


# In[21]:


plt.figure(figsize = (10, 5))
plt.hist(pollution['pm2.5'], bins = range(0, 1000, 50), ec = 'black')
plt.title('Distribution of PM2.5')
plt.xticks(range(0, 1000, 50))
plt.xlabel('Levels of pm2.5')
plt.ylabel('Count')
plt.show()


# This histogram shows that PM2.5, the target variable, is right-skewed. It appears that the air pollution was usually at the lower levels of 0 to 50 micrograms per cubic meter of air. The count of the levels of pm2.5 decreases as the level of pm2.5 increases. 
# 
# This is good because it shows that over a period of 5 years, there are generally lower levels of pm2.5 which is a good indication for people's health.

# ### DEWP

# In[22]:


# Understand the maximum and minimum values of the feature to understand possible values of bins
print(max(pollution['DEWP']))
print(min(pollution['DEWP']))


# In[23]:


plt.figure(figsize = (10, 5))
plt.hist(pollution['DEWP'], bins = range(-40, 30, 10), ec = 'black')
plt.title('Distribution of DEWP')
plt.xticks(range(-40, 30, 10))
plt.xlabel('DEWP Temperature')
plt.ylabel('Count')
plt.show()


# This histogram shows us the distribution of the variable DEWP (dew point temperature). While the distribution looks slightly left-skewed, we believe that it is not very left-skewed and it not of large concern. 
# 
# It does appear that most common dew point temperature is between 10 to 20 degrees celsius. However, the other bins of temperatures also occus pretty commonly. The least common bin(s) of temperature is -40 to -20 degrees of celsius.

# ### PRES

# In[24]:


# Understand the maximum and minimum values of the feature to understand possible values of bins
print(max(pollution['PRES']))
print(min(pollution['PRES']))


# In[25]:


plt.figure(figsize = (10, 5))
plt.hist(pollution['PRES'], bins = range(990, 1050, 10), ec = 'black')
plt.title('Distribution of PRES')
plt.xticks(range(990, 1050, 10))
plt.xlabel('Pressure (Pa)')
plt.ylabel('Count')
plt.show()


# This histogram shows that the distribution of the pressure is a normal distribution. This is good because it meets the normality requirement of linear regression. The most common values of pressure are 1000 to 1030 pascals.

# ### cbwd

# In[26]:


pollution['cbwd'].value_counts()


# In[27]:


pollution['cbwd'].value_counts().plot(kind = 'bar')
plt.title('Distribution of Wind Direction')
plt.xlabel('Wind Direction')
plt.ylabel('Count')
plt.xticks(rotation = 0)
plt.show()


# This bar graph shows that most of the observation records of pm2.5 levels have a wind direction of south east. The least common wind direction is north east.
# 
# There is a bar labeled 'cv' which, upon research, has been found to be equivalent to south west.
# 
# We can also vizualize this variable as a wafflechart to understand how the values of the wind direction vary:

# In[31]:


from pywaffle import Waffle

pollution_cbwd = pollution['cbwd'].value_counts()
pollution_cbwd = pollution_cbwd.to_frame('pollution_cbwd').reset_index()
pollution_cbwd.columns = ['Wind Direction', 'count']
pollution_cbwd


# In[40]:


waffle1 = plt.figure(FigureClass = Waffle,
                values = pollution_cbwd['count'],
                columns = 10,
                 rows = 8,
                figsize = (6,6),
                labels = list(pollution_cbwd['Wind Direction']),
                cmap_name="tab20",
                legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)})


# This waffle chart shows that mose of the values of the wind direction are either Southeast or Northwest. The least common values of wind direction is northeast with a very small proportion of the values being Northeast (approx. 11%).
# Approximately 35% of the values are Southeast and 31% of the values being Northwest. 
# 
# This visualization tells us the most common occurrence of the wind direction which also gives us various indications about the air qualities of Beijing. The wind direction allows us to understand the source of the pollutants. More information about this can be found by research but due to time constraints we have not been able to do this research.

# ### Iws

# In[ ]:


# Understand the maximum and minimum values of the feature to understand possible values of bins
print(max(pollution['Iws']))
print(min(pollution['Iws']))


# In[ ]:


plt.figure(figsize = (10, 5))
plt.hist(pollution['Iws'], bins = range(0, 600, 50), ec = 'black')
plt.title('Distribution of Iws')
plt.xticks(range(0, 600, 50))
plt.xlabel('Cumulated Wind Speed (meters per second)')
plt.ylabel('Count')
plt.show()


# This histogram shows that the most common value of the cumulated wind speed is between 0 and 50 meters per second. The frequency of each passing bin of wind speed decreases. This distribution is clearly right-skewed. 

# ### Is

# In[ ]:


# Understand the maximum and minimum values of the feature to understand possible values of bins
print(max(pollution['Is']))
print(min(pollution['Is']))


# In[ ]:


plt.figure(figsize = (10, 5))
plt.hist(pollution['Is'], bins = range(0, 30, 10), ec = 'black')
plt.title('Distribution of Is')
plt.xticks(range(0, 30, 10))
plt.xlabel('Cumulated Hours of Snow')
plt.ylabel('Count')
plt.show()


# This histogram describes the distribution of the cumulated hours of snow over the five years. While it can be said that the distribution is right-skewed, there is only one prominent bar and this histogram appears to be more like a single bar chart. The most common amount of cumulated hours of snow is between 0 and 10 hours.
# 
# We can decide to remove any values of Is that have values greater than 10 hours. This graph also shows that in Beijing, it snows for a cumulative amount of 10 hours most of the time.

# ### Ir

# In[ ]:


# Understand the maximum and minimum values of the feature to understand possible values of bins
print(max(pollution['Ir']))
print(min(pollution['Ir']))


# In[ ]:


plt.figure(figsize = (10, 5))
plt.hist(pollution['Is'], bins = range(0, 40, 10), ec = 'black')
plt.title('Distribution of Ir')
plt.xticks(range(0, 40, 10))
plt.xlabel('Cumulated Hours of Rain')
plt.ylabel('Count')
plt.show()


# This histogram describes the distribution of the cumulated hours of rain over the five years. While it can be said that the distribution is right-skewed, there is only one prominent bar and this histogram appears to be more like a single bar chart. The most common cumulated hours of rain is between 0 and 10 hours.
# 
# We can decide to remove any values of Ir that have values greater than 10 hours. This graph also shows that in Beijing, it rains for a cumulative amount of 10 hours most of the time.

# ---

# ## Correlation

# In[ ]:


corr = pollution.corr()
corr


# In[ ]:


plt.figure(figsize = (10,8))
sns.heatmap(corr, annot = True, cmap = 'coolwarm')


# Looking at the correlation table and matrix, it is clear that most variables are not highly correlated to each other; this is good because it indicates low multicollinearity. 
# 
# However, there are a few variables that show high correlation with each other:
# 1) DEWP and PRES: Has a high correlation of -0.78 (a negative correlation). This makes sense because the dew point temeprature will depend on the air pressure (according to its definition).
# 
# 2) DEWP and TEMP: Has a high correlation of 0.82 (a positive correlation). This high correlation makes sense because the dew point is related to the current temperature and humidity and other environmental and air characteristics. 
# 
# 3) TEMP and PRES: Has a high correlation of -0.83 (a negative correlation). This high correlation makes sense because the pressure of a gas is directly proportional to its temperature.
# 
# In relation to the target variable, it appears that no variable has a high correlation (greater tha 0.5 or less than -0.5). The variable that is most correlated with the target variable is Iws (cumulated wind speed). 

# Let us visualize these correlations with scatterplots to be able to tell a better story:

# In[ ]:


plt.scatter(x = pollution['DEWP'], y = pollution['PRES'], c = pollution['pm2.5'])
plt.title('DEWP v/s PRES colored by PM2.5 levels')
plt.xlabel('DEWP')
plt.ylabel('PRES')
plt.show()


# This scatterplot shows that there is a negative association between DEWP and PRES as we learned from the correlation matrix. It appears that as the dew point temperature increases, the air pressure decreases. This is interesting because it allows us to understand that an increasing temperature at which water condenses corresponds to a decreasing level fo air pressure. 
# 
# Further, it also shows that dew point is a measure of near-surface air quality. We have also found out that the dew condensation processes reduces the amount of pollutants near the surface. Thus, in a palce like China, if dew point is occurring at close to the average temperature, then the amount of pollutants must be less.

# In[ ]:


plt.scatter(x = pollution['DEWP'], y = pollution['TEMP'], c = pollution['pm2.5'])
plt.title('DEWP v/s TEMP colored by PM2.5 levels')
plt.xlabel('DEWP')
plt.ylabel('TEMP')
plt.show()


# This scatterplot is interesting because there is a clear cut boundary between the two variables. It appears that there is an increasing relation between the variables. This makes sense because as the current temperature increases, the dew point temperature will also increase. This information is useful because it will allow us to make a dependence relationship between these predictor variables and the level of pollution in Beijing. It appears that the slope between the two variable is close to 1 showing that temperature and dew point increase similarly.

# In[ ]:


plt.scatter(x = pollution['PRES'], y = pollution['TEMP'], c = pollution['pm2.5'])
plt.title('PRES v/s TEMP colored by PM2.5 levels')
plt.xlabel('PRES')
plt.ylabel('TEMP')
plt.show()


# This scatterplot shows that there is a clear negative relation between the two variables as was clear with the correlation matrix. This scatterplot shows that as the air pressure increases, the temperature decreases. Upon further investigation, it has been found that as the temperature rises, molecules become active which causes the denisty of air to reduce thus causing air pressure to lower. Further, when the air pressure is low, the air is still allowing pollution levels to rise indicating an inverse relationship between teh two (which is also shown by the correlation matrix).

# In[ ]:


plt.figure(figsize = (8,5))
plt.scatter(x = pollution['year'], y = pollution['pm2.5'], c = pollution['pm2.5'])
plt.title('Year PM2.5 levels')
plt.xlabel('YEAR')
plt.ylabel('PM2.5')
plt.show()


# This scatterplot shows that each year has nearly the same levels of pollution for each year. This shows that there is not any trends in pollution levels in comparison to the year (we cannot create a story of pollution levels over a yearly period). 
# 
# There is constant levels of pollution (pm2.5 amount) for each year, hence it does not appear that pollution levels is a temporal variable. This low correlation is also displayed in the correlation matrix.

# Let us look if there is a difference between months:

# In[ ]:


plt.figure(figsize = (8,5))
plt.scatter(x = pollution['month'], y = pollution['pm2.5'], c = pollution['pm2.5'])
plt.title('Monthly PM2.5 levels')
plt.xlabel('Month')
plt.ylabel('PM2.5')
plt.show()


# It appears that there appears to be a relationship between the month and the levels of PM2.5 in the air in Beijing. It appears that January has hte highest amounts of pollution and may has the least amount of pollution.
# 
# This makes sense because the Chinese New Year usually occurs in January or February which is a large cause of celebration in the country, especially the capital. The amount of pollution levels go down gradually ove erthe year and it appears that it starts steadily rising after the middle of the year which also makes sense because the number of celebrations increases as the year comes to a close (the information has been found by research). This low correlation is shown by the correlation matrix, but there is a higher correlation between month and level of PM2.5 than between year and PM 2.5.

# Let us look if the levels of PM 2.5 depends on the day of the month:

# In[ ]:


plt.figure(figsize = (8,5))
plt.scatter(x = pollution['day'], y = pollution['pm2.5'], c = pollution['pm2.5'])
plt.title('Daily PM2.5 levels')
plt.xlabel('Day')
plt.ylabel('PM2.5')
plt.show()


# This scatterplot shows that there is a very low correlation between the days and the level of pollution (also shown from the correlation matrix). There does appear to some days in the middle of the month that appear to have higher levels of pollution.
# 
# After some reseearch, we found that the Chinese New Year often is celebrated at the begninng of end of a month which explains this high level fo pollution. Further, constant transportation and factory emissions contribute to constant high levels of pollution which also explains the lack of correlation between the day and the level of pollution.

# ---

# Now that we have explored this dataset,
# ### Let us now move on to Data Cleaning
