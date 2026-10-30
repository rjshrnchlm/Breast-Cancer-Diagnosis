from keras import layers, models
import numpy as np
from keras.src.optimizers import Adam
from Evaluation import evaluation


def Model_CNN(train_data, train_target, test_data, test_target, SPE=None, HN=None, BS=None):
    if SPE is None:
        SPE = 5
    if HN is None:
        HN = 128
    if BS is None:
        BS = 32

    IMG_SIZE = 32
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 3))

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(HN, activation='relu'))  # 128
    model.add(layers.Dense(test_target.shape[-1], activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, train_target, epochs=30, batch_size=BS, steps_per_epoch=SPE, validation_data=(Test_X, test_target))
    pred = model.predict(Test_X)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = evaluation(test_target, pred)
    return Eval, pred

