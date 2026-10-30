import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from keras.src.optimizers import Adam
from Evaluation import evaluation


def Model_BILSTM(Train_Data, Train_Target, Test_Data, Test_Target, HN=None, EP=None, BS=None, LR=None):
    if HN is None:
        HN = 64
    if EP is None:
        EP = 50
    if BS is None:
        BS = 16
    if LR is None:
        LR = 0.001
    input = (1, 100)
    Train_Temp = np.zeros((Train_Data.shape[0], input[0], input[1]))
    for i in range(Train_Data.shape[0]):
        Train_Temp[i, :] = np.resize(Train_Data[i], (input[0], input[1]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input[0], input[1])

    Test_Temp = np.zeros((Test_Data.shape[0], input[0], input[1]))
    for i in range(Test_Data.shape[0]):
        Test_Temp[i, :] = np.resize(Test_Data[i], (input[0], input[1]))
    Test_X = Test_Temp.reshape(Test_Temp.shape[0], input[0], input[1])

    # Bi-LSTM model
    model = Sequential([
        Bidirectional(LSTM(HN, activation='tanh', return_sequences=False), input_shape=input),
        Dropout(0.2),
        Dense(Train_Target.shape[1], activation='linear')])
    model.compile(optimizer=Adam(learning_rate=LR), loss='mse', metrics=['mae'])
    model.fit(Train_X, Train_Target, epochs=EP, batch_size=BS, validation_data=(Test_X, Test_Target), verbose=1)
    pred_test_y = model.predict(Test_X.astype('float'), batch_size=BS, verbose=0)
    # pred_test_y[pred_test_y >= np.mean(pred_test_y)] = 1
    # pred_test_y[pred_test_y < np.mean(pred_test_y)] = 0
    # Eval = ClassificationEvaluation(Test_Target, pred_test_y)
    avg = np.mean(Test_Target)
    Test_Target[Test_Target >= avg] = 1
    Test_Target[Test_Target < avg] = 0
    Test_Target = Test_Target.astype('int')
    Eval = evaluation(Test_Target, pred_test_y)
    return Eval
