import streamlit as st
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D,BatchNormalization,Dropout,Flatten,Dense


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)

def preprocess_data():
    # Existing code to import necessary libraries, import the dataset, and preprocess the data
    # data=pd.read_csv("C:/Users/User/Desktop/creditcard.csv")
    data['Class'].value_counts()
    nonFraudData = data[data['Class']==0]
    fraudData = data[data['Class']==1]

    nonFraudData.shape, fraudData.shape

    nonFraudDataSample = nonFraudData.sample(fraudData.shape[0])

    nonFraudDataSample.shape

    balancedData = fraudData.append(nonFraudDataSample,ignore_index = True)
    balancedData

    balancedData['Class'].value_counts()

    x=balancedData.drop(['Class'], axis=1)
    y=balancedData.Class

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 42)

    sc=StandardScaler()
    xtrain=sc.fit_transform(xtrain)
    xtest=sc.fit_transform(xtest)

    xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
    xtest = xtest.reshape(xtest.shape[0],xtest.shape[1],1)

    ytrain=ytrain.to_numpy()
    ytest=ytest.to_numpy()

    #Building the Model: Convolutional Neural Network
    model = Sequential()

    model.add(Conv1D(32, kernel_size=2, activation = 'relu',input_shape = xtrain[0].shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv1D(64, kernel_size=2, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1,activation='sigmoid'))

    model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])

    model.fit(xtrain,ytrain, epochs=60, validation_data=(xtest,ytest))

    Prediction = model.predict(xtest)
    yPrediction = (yPrediction>0.5)

    return model


# Use the st.button function to create a button for the user to submit their data
if uploaded_file is not None and st.button("Predict"):
    model = preprocess_data()
    # Use the model to make a prediction
    prediction = model.predict(data)
    # Display the prediction to the user
    st.write("Prediction: ", prediction)
