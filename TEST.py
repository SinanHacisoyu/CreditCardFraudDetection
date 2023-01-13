import streamlit as st

def preprocess_data():
    # Existing code to import necessary libraries, import the dataset, and preprocess the data
    data=pd.read_csv('kaggle kernels output dhirajkumar612/credit-card-fraud-detection-different-models -p /path/to/dest')
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

# Use Streamlit to create a sidebar for the user to input their data
data = st.sidebar.text_input("Enter your data here")

# Use the st.button function to create a button for the user to submit their data
if st.button("Predict"):
    model = preprocess_data()
    # Use the model to make a prediction
    prediction = model.predict(data)
    # Display the prediction to the user
    st.write("Prediction: ", prediction)
