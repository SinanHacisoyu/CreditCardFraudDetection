import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D,BatchNormalization,Dropout,Flatten,Dense

sc=StandardScaler()

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)
   
def preprocess_data():
    data['Class'].value_counts()
    nonFraudData = data[data['Class']==0]
    fraudData = data[data['Class']==1]

    nonFraudDataSample = nonFraudData.sample(fraudData.shape[0])

    balancedData = fraudData.append(nonFraudDataSample,ignore_index = True)

    x=balancedData.drop(['Class'], axis=1)
    y=balancedData.Class

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 42)

    xtrain=sc.fit_transform(xtrain)
    xtest=sc.fit_transform(xtest)

    xtrain = xtrain.reshape(xtrain.shape[0],xtrain.shape[1],1)
    xtest = xtest.reshape(xtest.shape[0],xtest.shape[1],1)

    ytrain=ytrain.to_numpy()
    ytest=ytest.to_numpy()

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
    yPrediction = (Prediction>0.5)
    return balancedData

if uploaded_file is not None and st.button("Predict"):
    balancedData = preprocess_data()
    st.write("Number of data points in the balanced data: ", balancedData.shape[0])
    # preprocess the entire dataset
    x_all = data.drop(['Class'], axis=1)
    x_all = sc.transform(x_all)
    x_all = x_all.reshape(x_all.shape[0], x_all.shape[1], 1)
    # Use the model to make a prediction
    prediction = model.predict(x_all)
    # Display the prediction to the user
    st.write("Prediction: ", prediction)
