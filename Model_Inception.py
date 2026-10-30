from keras.src.applications.inception_v3 import InceptionV3
from keras.src.layers import Dense, GlobalAveragePooling2D
from keras.src.models import Model
import cv2 as cv
import numpy as np
from keras.src.optimizers import Adam
from Evaluation import evaluation


def Model_Inception_Feat(Data, Target, Act=None, HN=None, BS=None, sol=None):
    if sol is None:
        sol = 1024
    if BS is None:
        BS = 4
    if HN is None:
        HN = int(sol[0])
    if Act is None:
        Act = 1
    Activation = ['linear', 'relu', 'tanh', 'sigmoid', 'softmax', 'leaky relu']

    input_shape = (255, 255, 3)
    num_classes = Target.shape[-1]

    Datas = np.zeros((Data.shape[0], input_shape[0], input_shape[1] * input_shape[2]))
    for i in range(Data.shape[0]):
        Datas[i, :] = cv.resize(Data[i], (input_shape[1] * input_shape[2], input_shape[0]))
    Datas = Datas.reshape(Datas.shape[0], input_shape[0], input_shape[1], input_shape[2])

    # Load the InceptionV3 model (pre-trained on ImageNet)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(HN, activation=Activation[int(Act)])(x)  # 'relu'
    predictions = Dense(num_classes, activation='softmax')(x)  # 'softmax'
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Datas, Target, epochs=50, batch_size=BS, steps_per_epoch=10)
    pred = model.predict(Target)
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    Feats = feature_model.predict(Datas)
    Feature = np.asarray(Feats)
    return Feature


def Model_Inception(Train_Data, Train_Target, Test_Data, Test_Target, sol=None, Act=None, HN=None, BS=None):
    if BS is None:
        BS = 4
    if sol is None:
        sol = [1024]
    if HN is None:
        HN = int(sol[0])
    if Act is None:
        Act = 1

    Activation = ['linear', 'relu', 'tanh', 'sigmoid', 'softmax', 'leaky relu']

    input_shape = (255, 255, 3)
    num_classes = Test_Target.shape[-1]

    Feat1 = np.zeros((Train_Data.shape[0], input_shape[0], input_shape[1] * input_shape[2]))
    for i in range(Train_Data.shape[0]):
        Feat1[i, :] = cv.resize(Train_Data[i], (input_shape[1] * input_shape[2], input_shape[0]))
    train_data = Feat1.reshape(Feat1.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Feat2 = np.zeros((Test_Data.shape[0], input_shape[0], input_shape[1] * input_shape[2]))
    for i in range(Test_Data.shape[0]):
        Feat2[i, :] = cv.resize(Test_Data[i], (input_shape[1] * input_shape[2], input_shape[0]))
    test_data = Feat2.reshape(Feat2.shape[0], input_shape[0], input_shape[1], input_shape[2])

    # Load the InceptionV3 model (pre-trained on ImageNet)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(HN, activation=Activation[int(Act)])(x)  # 'relu'
    predictions = Dense(num_classes, activation='softmax')(x)  # 'softmax'
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(train_data, Train_Target, epochs=50, batch_size=BS, steps_per_epoch=10, validation_data=(test_data, Test_Target))
    pred = model.predict(test_data)
    Eval = evaluation(Test_Target, pred)
    return Eval, pred