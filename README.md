## Bitcoin-price-Detection-Using-Neural-Networks

A System to predict Bitcoin prices using LSTM neural networks

* Used Tensorflow as backend for keras.
* Used Keras to make LSTM neural network.
* Use of normalization to scale values to make neural network converge faster. 
* Used Matlplotlib for visualizations.

### Dataset 

Used dataset containing historical stock prices of Bitcoin list from Kaggle.  

CSV files for select bitcoin exchanges for the time period of Jan 2014 to December 2017, with minute to minute updates of OHLC (Open, High, Low, Close), Volume in BTC and indicated currency, and weighted bitcoin price. Timestamps are in Unix time. 

### Implementation Methodology

1.  Read the data , convert timestamp values to date and then group them according to their date.
2.  Split data into Training and Testing data
3.  Use MinMaxScaler from sklearn for Data preprocessing to feature scale data to fit in range (0,1) which helps NN converge faster.
4.  Initialising the RNN
5.  Add the input layer and the LSTM layer input_shape(x,y) x is number of time steps  and y is number of features.
6.  Use sigmoid activations function.
7.  Add the output layer
8.  Compile the RNN
9.  Fit the RNN to the training set
10. Transform the testing data using MinMaxScaler
11. Predict bitcoin price for given testing data
12. Take inverse transform of predicted data
13. Calculate root mean squared error.
14. Visualize the results.


