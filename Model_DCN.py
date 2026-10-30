import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    Add,
    Concatenate
)
from Evaluation import evaluation


def Model_DCN(train_data, train_target, test_data, test_target, SPE=None, EP=None, HN=None, BS=None, sol=None):

    if sol is None:
        sol = [1]
    if SPE is None:
        SPE = 5
    if HN is None:
        HN = 128
    if EP is None:
        EP = 30
    if BS is None:
        BS = 4

    # Number of output classes
    Classes = train_target.shape[-1]

    # Convert input data to NumPy arrays
    Train_X = np.asarray(train_data, dtype=np.float32)
    Test_X = np.asarray(test_data, dtype=np.float32)

    # Flatten input data if it has more than two dimensions
    if Train_X.ndim > 2:
        Train_X = Train_X.reshape(Train_X.shape[0], -1)

    if Test_X.ndim > 2:
        Test_X = Test_X.reshape(Test_X.shape[0], -1)

    # Replace NaN and infinite values
    Train_X = np.nan_to_num(Train_X)
    Test_X = np.nan_to_num(Test_X)

    # Normalize using training data statistics
    mean = np.mean(Train_X, axis=0, keepdims=True)
    std = np.std(Train_X, axis=0, keepdims=True)
    std[std == 0] = 1.0

    Train_X = (Train_X - mean) / std
    Test_X = (Test_X - mean) / std

    # Input feature dimension
    NUM_FEATURES = Train_X.shape[1]

    # DCN parameters
    CROSS_LAYERS = 3
    DEEP_UNITS = [128, 64, 32]
    DROPOUT_RATE = 0.2

    # Input layer
    inputs = Input(shape=(NUM_FEATURES,))

    # Original feature representation
    x0 = inputs
    cross = x0

    # -----------------------------------
    # Cross Network
    # -----------------------------------
    for i in range(CROSS_LAYERS):
        cross_output = Dense(
            NUM_FEATURES,
            activation=None,
            kernel_initializer="glorot_uniform",
            name=f"cross_dense_{i}"
        )(cross)

        cross_output = Add()([
            x0 * cross_output,
            cross
        ])

        cross = cross_output

    # Deep Network
    deep = x0

    for i, units in enumerate(DEEP_UNITS):
        deep = Dense(
            units,
            activation="relu",
            name=f"deep_dense_{i}"
        )(deep)

        deep = Dropout(DROPOUT_RATE)(deep)

    # Combine Cross and Deep Networks
    combined = Concatenate()([cross, deep])

    combined = LayerNormalization()(combined)

    combined = Dense(
        64,
        activation="relu"
    )(combined)

    combined = Dropout(DROPOUT_RATE)(combined)

    # Classification output
    outputs = Dense(
        Classes,
        activation="sigmoid"
    )(combined)

    # Build model
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="Deep_Cross_Network"
    )

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    model.fit(
        Train_X,
        train_target,
        epochs=2,
        batch_size=BS,
        validation_data=(Test_X, test_target),
        verbose=1
    )
    pred = model.predict(Test_X)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = evaluation(test_target, pred)
    return Eval, pred


