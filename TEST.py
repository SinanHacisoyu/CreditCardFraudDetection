#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Credit Card Fraud Detection
# Performing CNN on the Credit Card Dataset, in order to determine if the transactions are fraud or not.
# Library Used: TensorFlow, Sklearn, Keras


# In[2]:


# Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv1D, MaxPool1D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# In[3]:


# Import the dataset
data=pd.read_csv('C:/Users/User/Desktop/archive4.zip')
data.head()


# In[4]:


#To determine the Data types of the features
data.dtypes


# In[5]:


# Check the shape of the dataset
data.shape
# There are 31 columns here, and the final column contains the Class target variable.


# In[6]:


# To get all the details
data.describe()


# In[7]:


# To determine if there is any null values present in the dataset
data.isnull().sum()
#Can be seen that there is no null value in the dataset


# In[8]:


#To get all the information about all the features
data.info()
#Can be determine by looking at the dataset and the information that our target column has values consisting 0 and 1.


# In[9]:


# Determine the number of values in the "class" that have values of 0 or 1 that are present.
data['Class'].value_counts()


# In[10]:


# There are 284315 transactions 
# 492 transactions are fraud only.
# Now, balancing the dataset.


# In[11]:


# Separating fraud and non-fraud rows
nonFraudData = data[data['Class']==0]
fraudData = data[data['Class']==1]

nonFraudData.shape, fraudData.shape


# In[12]:


# Selecting the 492 non-fraud entries from the dataframe 
nonFraudDataSample = nonFraudData.sample(fraudData.shape[0])

nonFraudDataSample.shape


# In[13]:


# Now there is balanced dataset: rows 492 fraud , 492 non-fraud
balancedData = fraudData.append(nonFraudDataSample,ignore_index = True) 
balancedData


# In[14]:


# Now the data has been balanced and combined.
# Check again the value counts
balancedData['Class'].value_counts()


# In[15]:


# now dividing the dataframe into dependent and independent varaible
x=balancedData.drop(['Class'], axis=1)
y=balancedData.Class

# check the shape
x.shape, y.shape


# In[16]:


# Splitting in to Train and test dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 42)

# Check the shape again
print(xtrain.shape,xtest.shape,ytrain.shape,ytest.shape)


# In[17]:


# Scaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)


# In[18]:


# Converting 2D dataset into 3D For CNN prediction
# xtrain = xtrain.to_numpy()
# xtest = xtest.to_numpy()


# In[19]:


xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
xtest = xtest.reshape(xtest.shape[0],xtest.shape[1],1)

# Cheking the shape
xtrain.shape, xtest.shape


# In[20]:


# ytrain, ytest are in series, converting the same into a numpy array
ytrain=ytrain.to_numpy()
ytest=ytest.to_numpy()


# In[21]:


#Building the Model: Convolutional Neural Network


# In[22]:


# Importing model
model = Sequential()

# layers:

# 1) First
# The model starts with adding a 1D convolutional layer with 32 filters,
# kernel size of 2, a ReLU activation function, and input shape of the xtrain data.
# It then applies batch normalization which normalize the activations of the previous layer at each batch.
# The code then applies dropout with a rate of 0.2 to prevent overfitting.
model.add(Conv1D(32, kernel_size=2, activation = 'relu',input_shape = xtrain[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))


# In[23]:


# 2) Second
# The code then adds another 1D convolutional layer with 64 filters,
# kernel size of 2, a ReLU activation function, and applies batch normalization and dropout 
# with a rate of 0.5 again to prevent overfitting.
model.add(Conv1D(64, kernel_size=2, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5)) 


# In[24]:


# 3) building Artificial neural network (ANN)
# Then the code flattens the output of the previous layer and add a dense layer with 64 units 
# and ReLU activation function, followed by dropout with a rate of 0.5 to prevent overfitting.
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))


# In[25]:


# 4) Last
# The code adds the last dense layer with 1 unit and a sigmoid activation function.
# This layer is used as the output layer, and the activation function (sigmoid) is used for binary classification problems.
model.add(Dense(1,activation='sigmoid'))


# In[26]:


#It shows the layers of the model, the number of parameters in each layer, 
#the shape of the output of each layer, and the total number of parameters in the model.
#This is useful for understanding the structure of the model and for debugging.
model.summary()


# In[27]:


# compiling the model
# We can change the lr during the training process.
model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])


# In[28]:


#It trains the model on the training data (X_train and y_train), for 20 epochs and also passing 
#the test data (X_test and y_test) as validation_data to evaluate performance after each epoch.
model.fit(xtrain,ytrain, epochs=60, validation_data=(xtest,ytest))


# In[30]:


# Making the Prediction on a test set of data (xtest) and Evaluating the model
yPrediction = model.predict(xtest)
yPrediction = (yPrediction>0.5)#Converts the predicted probability values into binary values by using a threshold of 0.5.


# In[31]:


from sklearn.metrics import roc_auc_score

# Get the predicted probability values for the positive class (class 1)
yPredictionProb = model.predict(xtest)[:,0]
# Calculate the ROC AUC score
rocAUC = roc_auc_score(ytest, yPredictionProb)
print("ROC AUC: {:.3f}".format(rocAUC))


# In[33]:


# evaluating the accuracy score and confusion matrix
# Checking the accuracy
import seaborn as sns
import matplotlib.pyplot as plt
# Plotting the confusion matrix
confMatrix = confusion_matrix(ytest, yPrediction)
sns.heatmap(confMatrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()


# In[34]:


from sklearn.metrics import precision_recall_fscore_support as score


# In[35]:


precision, recall, fscore, support = score(ytest, yPrediction)
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('Fscore: {}'.format(fscore))
print('Support: {}'.format(support))

