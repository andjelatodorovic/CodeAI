#!/usr/bin/env python
# coding: utf-8

# <h1><center>Africa Soil Property Prediction</center></h1>

# #### Import modules

# In[1]:


import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn import preprocessing
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import seaborn as sns
import statistics
pd.set_option('display.max_columns', 1157)


# #### Helper functions

# In[3]:


def mcrmse(y_true, y_pred):
    rmse_vals = []
    for i in range(5):
        rmse_vals.append(sqrt(mean_squared_error(y_test.values[:, i], y_pred[:, i])))
    mcrmse = statistics.mean(rmse_vals)
    return (rmse_vals, mcrmse)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# #### Load dataset

# In[4]:


# Load train dataset
train_df = pd.read_csv("~/Desktop/code/data/raw/training.csv")

# Load test dataset
test_df = pd.read_csv("~/Desktop/code/data/raw/sorted_test.csv")

train_df.head()


# #### Inserting a value of spectrometer data

# In[44]:


slice0 = train_df.iloc[:,0:3578]


# In[45]:


slice0


# In[5]:


slc = [3579,3580,3581,3582,3583,3584,3585,3586,3587,3588,3589,3590,3591,3592,3593,3594,3595,3596,3597,3598,3599]
train_df.values[:, slc]


# In[6]:


slice1 = pd.DataFrame(train_df.values[:, slc], train_df.index, train_df.columns[slc])


# In[7]:


slice1.to_csv(index="false")


# In[8]:


slice1


# In[9]:


slice2 = train_df.iloc[:,:-21]


# In[10]:


slice2


# In[11]:


slice2.to_csv(index="false",sep = ',')


# #### Data cleaning

# In[12]:


train_df = train_df.drop(['PIDN'], axis=1)
test_df = test_df.drop(['PIDN'], axis=1)


# In[13]:


#for i in range (1157)
    #sample_bym = slice1.iloc[0, 0:12].values


# In[14]:


filtered_data = []
for i in range (train_df.shape[0]):
    tmp=train_df.iloc[i,0:3578]
    tmp_savgol = savgol_filter(tmp, 51, 5)
    filtered_data.append(tmp_savgol)


# In[15]:


np.array(filtered_data).shape


# In[16]:


indices = np.array(filtered_data)
columns = np.array(filtered_data[0])
results_filtered = pd.DataFrame(data=np.array(filtered_data), index = indices, columns = columns)


# #### Spectrum analysis

# In[18]:


spectrum_sample = train_df.iloc[0, 0:3578].values

# Spectrum smoothing
y_savgol = savgol_filter(spectrum_sample, 51, 5)
y_conv_avg = smooth(spectrum_sample, 50)


# #### Savitzki- Golay filtering

# In[19]:


plt.figure(figsize=(15,10))
plt.plot(spectrum_sample, label="No filter")
plt.plot(y_savgol, color="red", linestyle='dotted', label="Savitzky-Golay")
plt.plot(y_conv_avg, color="green", linestyle='dashed', label="Convolution Box filter")
plt.xlabel("Wavelength (nm)", fontsize=15)
plt.ylabel("Reflectance", fontsize=15)
plt.title("Soil infrared spectroscopy)", fontsize=20)
plt.legend(loc="upper left", fontsize=18)


# #### Split data into features and targets

# In[20]:


# Define labels
target_labels = ['Ca', 'P', 'pH', 'SOC', 'Sand']

# Train dataset
train_features_df = train_df.iloc[:,:-5]
train_targets_df = train_df[target_labels]

# Test challenge dataset
test_features_df = test_df

# Test dataset
# test_features_df = test_df.iloc[:,:-5]
# test_targets_df = train_df[target_labels]


# #### Cathegorical data encoding

# In[21]:


le = LabelEncoder()

# Encode train dataset
train_features_df["Depth_encoded"] = le.fit_transform(train_features_df["Depth"])
train_features_df = train_features_df.drop(["Depth"], axis=1)

# Encode test dataset
le = LabelEncoder()
test_features_df["Depth_encoded"] = le.fit_transform(test_features_df["Depth"])
test_features_df = test_features_df.drop(["Depth"], axis=1)


# #### Split data into train and test datasets

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(train_features_df, train_targets_df, test_size=0.2, random_state=0)


# #### Data scaling

# In[23]:


# scaler = preprocessing.MinMaxScaler()

# Use StandardScaler, it gives better results for MLP, SVM!
scaler = preprocessing.StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# #### Create models

# In[24]:


results = []


# ##### - Gradient Boosting Regression

# In[25]:


model_GB = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))
y_pred_gb = model_GB.fit(X_train_scaled, y_train).predict(X_test_scaled)
mcmrse_GB = mcrmse(y_test, y_pred_gb)
results.append(mcmrse_GB)
print("MCRMSE: ", mcmrse_GB)


# ##### - Random Forest Regression

# In[26]:


model_RF = MultiOutputRegressor(RandomForestRegressor(random_state=0))
y_pred_rf = model_RF.fit(X_train_scaled, y_train).predict(X_test_scaled)
mcrmse_RF = mcrmse(y_test, y_pred_rf)
results.append(mcrmse_RF)
print("MCRMSE: ", mcrmse_RF)


# ##### - Linear Regression

# In[27]:


model_LR = MultiOutputRegressor(LinearRegression())
y_pred_lr = model_LR.fit(X_train_scaled, y_train).predict(X_test_scaled)
mcrmse_LR = mcrmse(y_test, y_pred_lr)
results.append(mcrmse_LR)
print("MCRMSE: ", mcrmse_LR)


# ##### - Ridge Regression

# In[28]:


model_Ridge = MultiOutputRegressor(Ridge())
y_pred_ridge = model_Ridge.fit(X_train_scaled, y_train).predict(X_test_scaled)
mcrmse_Ridge = mcrmse(y_test, y_pred_ridge)
results.append(mcrmse_Ridge)
print("MCRMSE: ", mcrmse_Ridge)


# #### - MLP Regression

# In[29]:


model_MLP = MultiOutputRegressor(MLPRegressor(random_state=0))
y_pred_MLP = model_MLP.fit(X_train_scaled, y_train).predict(X_test_scaled)
mcrmse_MLP = mcrmse(y_test, y_pred_MLP)
results.append(mcrmse_MLP)
print("MCRMSE: ", mcrmse_MLP)


# #### - SVM Regression

# In[32]:


model_SVR = MultiOutputRegressor(SVR())
y_pred_SVR = model_SVR.fit(X_train_scaled, y_train).predict(X_test_scaled)
mcrmse_SVR = mcrmse(y_test, y_pred_SVR)
results.append(mcrmse_SVR)
print("MCRMSE: ", mcrmse_SVR)


# In[33]:


results = []
results.append(mcmrse_GB[0])
results.append(mcrmse_RF[0])
results.append(mcrmse_LR[0])
results.append(mcrmse_Ridge[0])
results.append(mcrmse_MLP[0])
results.append(mcrmse_SVR[0])


# In[34]:


indices = [" Gradient Boosting Regression", "Random Forest Regression", "Linear Regression", "Ridge Regression", "MLP Regression", "SVM Regression"]
columns = ['Ca', 'P', 'pH', 'SOC', 'Sand']

results_df = pd.DataFrame(data=np.array(results), index=indices, columns=columns)


# results_df = pd.DataFrame(data=np.array(results))


# In[35]:


results_df.head(6)


# In[36]:


results_df2 = results_df.transpose()


# In[37]:


type(results_df2)


# In[38]:


0.4476500411683135, 0.9644901027554718, 0.49204065304193234, 0.5211392510360906, 0.4459615792502725


# In[41]:


results_df2.plot(kind="bar")


# In[43]:


#import plotly.plotly as py
#import cufflinks as cf
#import pandas as pd
#import numpy as np

#cf.set_config_file(offline=False, world_readable=True, theme='ggplot')

#results_df.iplot(kind='bar')


# In[ ]:


# mcmrse_GB = mcrmse(y_test, y_pred_gb)
# print("MCRMSE: ", mcmrse_GB)
# x = ['Ca', 'P', 'pH', 'SOC', 'Sand']
# y = mcmrse_GB[0]

# sns.axes_style('white')
# sns.set_style('white')

# sns.barplot(x,y)


# In[ ]:





# In[ ]:




