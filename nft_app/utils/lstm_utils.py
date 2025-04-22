import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def predict_price_lstm(nft_name, data):
    if nft_name not in data['asset.name'].values:
        return None

    look_back = 1
    nft_data_selected = data[data['asset.name'] == nft_name].sort_values(by='sales_datetime')['price_in_ether'].values
    nft_data_selected = nft_data_selected.reshape(-1, 1)

    if len(nft_data_selected) <= look_back + 2:
        return "Insufficient data for LSTM prediction"

    train_size = int(len(nft_data_selected) * 0.7)
    train, test = nft_data_selected[0:train_size, :], nft_data_selected[train_size:len(nft_data_selected), :]

    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=0)

    testPredict = model.predict(testX)
    testPredict = scaler.inverse_transform(testPredict)

    return testPredict[-1][0]
