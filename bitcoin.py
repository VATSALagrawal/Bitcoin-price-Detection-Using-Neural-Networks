import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# read the data , convert timestamp values to date and then group them according to their date
df = pd.read_csv('coinbaseUSD.csv')
df['Date'] = pd.to_datetime(df['Timestamp'],unit='s').dt.date
#df=pd.read_csv('BTC-USD.csv')
group = df.groupby('Date')
Price = group['Close'].mean()

# number of days defined to test model
days_to_predict = 90
df_train= Price[:len(Price)-days_to_predict]
df_test= Price[len(Price)-days_to_predict:]

print("Training data")
print(df_train.head())
print("Testing data")
print(df_test.head())

# Data preprocessing to feature scale data to fit in range (0,1) which helps NN converge faster
training_set = df_train.values
training_set = np.reshape(training_set, (len(training_set), 1))
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
X_train = np.reshape(X_train, (len(X_train), 1, 1))

#print(X_train[0:5])
#print(y_train[0:5])

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
model = Sequential()

# Adding the input layer and the LSTM layer
# input_shape(x,y) x is number of time steps  and y is number of features
model.add(LSTM(units = 7, activation = 'sigmoid', input_shape = (None, 1)))
#model.add(Dropout(0.2))

#model.add(LSTM(units = 5, activation = 'sigmoid', input_shape = (None, 1),return_sequences=False))

# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, batch_size = 5, epochs = 100)


test_set = df_test.values
inputs = np.reshape(test_set, (len(test_set), 1))
inputs = sc.transform(inputs)
print("inputs before reshape ",inputs[0:5])
inputs = np.reshape(inputs, (len(inputs), 1, 1))
print("inputs after reshape ",inputs[0:5])

#input to every LSTM layer must be 3 dimentional
predicted_BTC_price = model.predict(inputs)
predicted_BTC_price = sc.inverse_transform(predicted_BTC_price)

#calculating root mean squared error
valid = np.reshape(test_set, (len(test_set), 1))
rms=np.sqrt(np.mean(np.power((valid-predicted_BTC_price),2)))
print("root mean error is :",rms)

# Visualising the results
plt.figure(figsize=(25,15), dpi=80, facecolor='w', edgecolor='k')
ax = plt.gca()
plt.plot(test_set, color = 'red', label = 'Real BTC Price')
plt.plot(predicted_BTC_price, color = 'green', label = 'Predicted BTC Price')
plt.title('Bitcoin Price Prediction', fontsize=25)
df_test = df_test.reset_index()
x=df_test.index
labels = df_test['Date']
plt.xticks(x, labels, rotation = 'vertical')
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(10)
plt.xlabel('Time', fontsize=30)
plt.ylabel('BTC Price(USD)', fontsize=20)
plt.legend(loc=2, prop={'size': 20})
plt.show()
