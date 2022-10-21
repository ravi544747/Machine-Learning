#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# **In Algerian dataset , considering the Temperature as dedendent variable and rest of all except calssess column as Independent varibales, Need to predict the temperatues based on other Independent varibales which are shown in below**

# * **Day** :  * **Month**: * **Year**: (DD/MM/YYYY) Day, month ('june' to 'september') , year (2012) Weather data observations
# * **Temperature**: temperature noon (temperature max) in Celsius degrees: 22 to 42
# * **RH**:   Relative Humidity in %: 21 to 90
# * **WS**:   Wind speed in km/h: 6 to 29
# * **Rain**: Total day in mm: 0 to 16.8
# * **FFMC**: Fine Fuel Moisture Code (FFMC) index from the FWI system: 28.6 to 92.5
# * **DMC**:  Duff Moisture Code (DMC) index from the FWI system: 1.1 to 65.9
# * **DC**:   Drought Code (DC) index from the FWI system: 7 to 220.4
# * **ISI**:  Initial Spread Index (ISI) index from the FWI system: 0 to 18.5
# * **BUI**:  Buildup Index (BUI) index from the FWI system: 1.1 to 68
# * **FWI**:   Fire Weather Index (FWI) Index: 0 to 31.1
# * **CLASSES**:Classes: two classes, namely "fire" and "not fire"
# * **region**: it provides the s2 pecified region namely  Bejaia and Sidi-Bel Abbes

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[ ]:





# In[2]:


# importing data skipping unnecessary rows and adding a seperate column based on the area of forest fires Bejaia and sidi-Bel as column names
# Bejaia Region Dataset- Bejaia, Sidi-Bel Abbes Region Dataset- Sedi-bel 


data=pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv", skiprows=[0,124,125,126], index_col=None)


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.describe()


# In[7]:


data.columns


# **there is extra space after the column- Classess , need add while dropping the column.**

# In[8]:


# dropping the column -CLassess
data=data.drop(['Classes  '],axis=1)


# In[9]:


data.head()


# In[10]:


data.dtypes


# There 2 columns "DC" and "FWI" are in object types , need to convert those to float64
#     **DC --Drought Code (DC) index from the FWI system: and  FWI--Fire Weather Index (FWI) Index: 0 to 31.1 have data type as objects both of them need to be converted into Numerical features as the data inside resembles numerical data**

# In[11]:


# convering FWI and "DC " to floating point variables

## converting DC and FWI to nemerical feature

# we have observed a "14.6 9" in DC and fire in FWI in the same row we are going to drop this line.

data.drop(data[data['DC'] =='14.6 9'].index, inplace = True)

data[['DC','FWI']] = data[['DC','FWI']].astype(str).astype(float)


# In[12]:


data.dtypes


# In[13]:


data.describe()


# In[14]:


data.info()


# In[15]:


# moving the Temperature column to last:

data_new=data.loc[:,data.columns!='Temperature']
data_new['Temperature']=data['Temperature']


# In[16]:


data.info()


# In[17]:


# Temperature column is moved to last
data_new.info()


# ### Preparing the data 

# In[18]:


# finding out null values in a data
data_new.isnull().sum()


# In[19]:


data_new.corr()


# In[20]:


(data_new.corr()>0.95).sum()


# * since year, day and month are discrete variables, the data is collected for the year 2012 we can drop the column year as it will not impact the temperature
# * DMC and BUI columns are 98 % correlated we can drop either one of the colums , we drop DMC

# In[21]:


# droping the Year and DMC column:
data_new=data_new.drop(['year','DMC'],axis=1)


# In[22]:


data_new.head()


# In[23]:


data_new.corr()


# In[24]:


sns.pairplot(data_new)


# In[25]:


plt.scatter( data_new['FFMC'],data_new['Temperature'],color='r')
plt.ylabel('Temperature')
plt.xlabel('FFMC')


# In[26]:


sns.regplot(x='FFMC', y="Temperature",data=data_new)
# shaded line: ridge and lasso 


# In[27]:


plt.figure(figsize = (20,20))
sns.heatmap(data_new.corr(), cmap="CMRmap", annot=True, linewidths=1,square=True, cbar=True)
plt.show()


# In[28]:


## INDEPENDENT and dependent features
X= data_new.iloc[:,:-1]
Y=data_new.iloc[:,-1]


# In[29]:


X


# In[30]:


Y


# ### splitting the data to train and test

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)


# In[33]:


X_train


# In[34]:


X_test


# ### Feature Engineering

# In[35]:


#Feature Scaling, Stadadize the dataset
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[36]:


scaler


# In[37]:


X_train=scaler.fit_transform(X_train)


# In[38]:


X_test=scaler.transform(X_test)


# In[39]:


X_train


# In[40]:


X_test


# ### Model Training

# In[41]:


from sklearn.linear_model import LinearRegression


# In[42]:


regression = LinearRegression()


# In[43]:


regression


# In[44]:


regression.fit(X_train, Y_train)


# ### Print the coefficients and intercepts

# In[45]:


print("Intercept", regression.intercept_)
print("coefficients",regression.coef_)


# ### prediction for train data

# In[59]:


Y_reg_pred_train=regression.predict(X_train)


# In[60]:


Y_reg_pred_train


# In[61]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(Y_train,Y_reg_pred_train))
print(mean_absolute_error(Y_train,Y_reg_pred_train))
print(np.sqrt(mean_squared_error(Y_train, Y_reg_pred_train)))


# In[62]:


from sklearn.metrics import r2_score
score=r2_score(Y_train,Y_reg_pred_train)
print(score)


# In[63]:


## Adjusted R square error

1 - (1-score)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)


# In[ ]:





# ### prediction for test data

# In[46]:


Y_reg_pred=regression.predict(X_test)


# In[47]:


Y_reg_pred


# ### Assumption of linear Regression

# In[48]:


# first Assumption , relation between Y actual and Y predicted 
plt.scatter(Y_test,Y_reg_pred)
x=plt.xlabel(" Test Truth Data")
y=plt.ylabel("Test Predicted data")


# In[49]:


##  2: Residuals:
residuals =Y_test-Y_reg_pred


# In[50]:


residuals


# In[51]:


sns.distplot(residuals)


# In[52]:


## Scatter plot with Redictions and residuals
## Uniform distribution

plt.scatter(Y_reg_pred, residuals)


# ### Performance Metrix

# In[53]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print(mean_squared_error(Y_test,Y_reg_pred))
print(mean_absolute_error(Y_test,Y_reg_pred))
print(np.sqrt(mean_squared_error(Y_test, Y_reg_pred)))


# ### R square and Adjusted R square

# In[54]:


from sklearn.metrics import r2_score
score=r2_score(Y_test,Y_reg_pred)
print(score)


# In[55]:


## Adjusted R square error

1 - (1-score)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)


# **the obtained model is 59.7% efficient in the train data and 51.9 efficeint with the test data**

# In[ ]:




