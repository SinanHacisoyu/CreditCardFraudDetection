import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D,BatchNormalization,Dropout,Flatten,Dense

sc=StandardScaler()

uploaded_file = st.file_uploader("Select your dataset to perform Credit Card Fraud Detection ")
if uploaded_file is not None:
  data = pd.read_csv(uploaded_file)
   
def preprocess_data():
  if 'Class' not in data:
        st.write("Error: 'Class' column not found in the dataset")
        return None, None, None, None, None
    data['Class'].value_counts()
    nonFraudData = data[data['Class']==0]
    fraudData = data[data['Class']==1]

    nonFraudDataSample = nonFraudData.sample(fraudData.shape[0])

    balancedData = fraudData.append(nonFraudDataSample,ignore_index = True)

    x=balancedData.drop(['Class'], axis=1)
    y=balancedData.Class

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, random_state = 42)

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

    model.fit(xtrain,ytrain, epochs=40, validation_data=(xtest,ytest))

    Prediction = model.predict(xtest)
    yPrediction = (Prediction>0.5)

    return model,xtrain, xtest, ytrain, ytest
  

if uploaded_file is not None and st.button("Predict"):
    model, xtrain, xtest, ytrain, ytest = preprocess_data()
    # preprocess the entire dataset
    x_all = data.drop(['Class'], axis=1) # Removing the 'Class' column from the dataframe.
    x_all = sc.transform(x_all) # Applying the StandardScaler to the data
    x_all = x_all.reshape(x_all.shape[0], x_all.shape[1], 1) # reshaping the data to match the input shape expected by the model
    
    # Use the model to make a prediction
    prediction = model.predict(x_all)
    
    # display the number of dataset
    st.write("Number of data points in the dataset: ", len(x_all))
    
    # display the number of fraud data points
    fraud_indices = [i for i in range(len(x_all)) if prediction[i] == 1] # Creates a list of indices of the data points that have been predicted as fraud by the model.
    fraud_data = data.iloc[fraud_indices]
    st.write("Predicted Fraud Data Points:") # Selects the rows of the original data (data) that correspond to the indices in the fraud_indices list
    st.dataframe(fraud_data) # Display the dataframe containing the predicted fraud data points to the user
    number_of_fraud_data = len(fraud_indices) # Calculates the number of fraud data points by getting the length of the fraud_ind
    st.write("Number of data points predicted as fraud: ", number_of_fraud_data)
    
    # display the loss and accuracy
    loss, accuracy = model.evaluate(xtest, ytest, verbose=0)
    st.write("loss: ", loss)
    st.write("accuracy: ", accuracy)
