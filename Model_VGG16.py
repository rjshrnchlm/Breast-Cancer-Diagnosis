from keras.src.models import Sequential
import numpy as np
from keras.src.layers import Conv2D, Flatten, Dense
from keras.src.optimizers import Adam
from keras.src.models import Model
from tensorflow.keras.layers import MaxPool2D
from Evaluation import evaluation


# from tf_keras.src.layers import MaxPool2D


def VGG_16(weights_path=None, num_of_class=None,HN=None):
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=HN, activation="relu"))
    model.add(Dense(units=num_of_class, activation="softmax"))
    weight = np.asarray(model.get_weights()[30])
    return model


def Model_VGG16(Train_Data, Train_Tar, Test_Data, Test_Tar, EP=None, BS=None, HN=None, sol=None):
    if BS is None:
        BS = 4
    if HN is None:
        HN = 4096
    if sol is None:
        sol = [50]
    if EP is None:
        EP = int(sol[0])

    IMG_SIZE = [32, 32, 3]

    Train1 = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Train1[i, :, :] = np.resize(Train_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Train = Train1.reshape(Train1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Test1 = np.zeros((Test_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Test_Data.shape[0]):
        Test1[i, :, :] = np.resize(Test_Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Test = Test1.reshape(Test1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    model = VGG_16(num_of_class=Train_Tar.shape[1], HN=HN)
    model.summary()
    # model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=Train, y=Train_Tar, epochs=EP, batch_size=BS, steps_per_epoch=5)
    pred = model.predict(Test)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = np.asarray(pred).astype('int')
    Eval = evaluation(Test_Tar, pred)
    return Eval, pred


def Model_VGG16_Feat(Data, Target, EP=None, BS=None, HN=None, sol=None):
    if BS is None:
        BS = 4
    if HN is None:
        HN = 4096
    if EP is None:
        EP = 50
    if sol is None:
        sol = [5]
    IMG_SIZE = [32, 32, 3]

    Datas = np.zeros((Data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(Data.shape[0]):
        Datas[i, :, :] = np.resize(Data[i], (IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    Datas = Datas.reshape(Datas.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    model = VGG_16(num_of_class=Target.shape[1], HN=HN)
    model.summary()
    # model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=Datas, y=Target, epochs=int(sol[0]), batch_size=BS, steps_per_epoch=5)
    pred = model.predict(Datas)
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    Feats = feature_model.predict(Datas)
    Feature = np.asarray(Feats)

    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = np.asarray(pred).astype('int')
    Eval = evaluation(Target, pred)
    return Feature, Eval, pred
