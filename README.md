# CreditCardFraudDetection

Credit Card Fraud Detection Performing CNN on the Credit Card Dataset, in order to determine if the transactions are fraud or not. Library Used: TensorFlow, Sklearn, Keras.

The reason for using a CNN is because it is well-suited for image and time-series data, which is what the credit card transactions data is. The data is treated as a time-series data where each transaction has several features, and the CNN architecture is able to learn the relevant features and patterns in the data.

In the code, the CNN model consists of several convolutional and dense layers that are used to learn the relationships between the features and the fraud/non-fraud labels. The model uses the 'relu' activation function in the Conv1D layers and 'sigmoid' activation function in the output layer, and the 'adam' optimizer and 'binary_crossentropy' loss function for training the model. The model is trained on a balanced dataset with equal number of fraud and non-fraud data points and is evaluated on a validation set.

The trained model is then used to make predictions on the entire dataset and the predicted fraud data points are displayed to the user. The performance of the model is evaluated in terms of accuracy and loss.

Streamlit.io is used for the API part. The Test.py file does the same as the CreditCardFraudDetection.ipynb file and is used for the interface.

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Api:
https://sinanhacisoyu-creditcardfrauddetection-test-e7497s.streamlit.app/

