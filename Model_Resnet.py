import numpy as np
from keras import Sequential
from keras.src.applications.resnet import ResNet50
from keras.src.layers import Dense
from Evaluation import evaluation
from keras.src.models import Model


def Model_Resnet(train_data, train_target, test_data, test_target, BS=None, HN=None, sol=None):
    if sol is None:
        sol = [0]
    if BS is None:
        BS = 4
    if HN is None:
        HN = 5
    Activation = ['linear', 'relu', 'tanh', 'sigmoid', 'softmax']
    Classes = test_target.shape[-1]
    inputs = (32, 32, 3)

    IMG_SIZE = 32
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, ( IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=inputs))
    base_model.add(Dense(units=HN, activation=Activation[int(sol[0])]))
    base_model.add(Dense(units=Classes, activation='linear'))  # units=train_target.shape[1] HN
    base_model.compile(loss='binary_crossentropy', metrics=['acc'])
    base_model.fit(Train_X, train_target, epochs=2, batch_size=BS, validation_data=(Test_X, test_target))
    pred = base_model.predict(Test_X)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = evaluation(test_target, pred)
    return Eval, pred


def Model_Resnet_Feat(Data, Target, BS=None, HN=None, sol=None):
    if sol is None:
        sol = [1]
    if BS is None:
        BS = 4
    if HN is None:
        HN = 5
    Classes = Target.shape[-1]
    inputs = (32, 32, 3)

    IMG_SIZE = 32
    Datas = np.zeros((Data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMG_SIZE * IMG_SIZE, 3))
        Datas[i] = np.reshape(temp, ( IMG_SIZE, IMG_SIZE, 3))

    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=inputs))
    base_model.add(Dense(units=HN, activation='linear'))
    base_model.add(Dense(units=Classes, activation='linear'))  # units=train_target.shape[1] HN
    base_model.compile(loss='binary_crossentropy', metrics=['acc'])
    base_model.fit(Datas, Target, epochs=2, batch_size=BS, validation_data=(Datas, Target))
    pred = base_model.predict(Datas)
    feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    Feats = feature_model.predict(Datas)
    Feature = np.asarray(Feats)
    return Feature