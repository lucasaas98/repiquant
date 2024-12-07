# Standard Library
from time import time

# Third party dependencies
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import callbacks, layers

# Current project dependencies
import helper as h


def load_data(short=False):
    # Define the list of tickers to use
    all_tickers = h.get_reasonable_tickers()
    interval = "5min"

    all_data = []

    # Load dataset from CSV files
    for ticker in all_tickers:
        scaled_labeled_50_bars = pd.read_csv(h.get_scaled_labeled(ticker, interval, short=short), index_col=0)

        scaled_labeled_50_bars = scaled_labeled_50_bars.drop(
            ["avg_stoploss", "avg_bars_in_market", "avg_takeprofit", "avg_return_pct"], axis=1
        )
        all_data.append(scaled_labeled_50_bars)

    # Concatenate all dataframes into one dataframe
    return pd.concat(all_data)


def split_data(train_data, short=False):
    if not short:
        train_data.loc[train_data["label"] == "very good", "label"] = 2
        train_data.loc[train_data["label"] == "good", "label"] = 1
        train_data.loc[train_data["label"] == "noop", "label"] = 0
    else:
        train_data.loc[train_data["label"] == "very bad", "label"] = 2
        train_data.loc[train_data["label"] == "bad", "label"] = 1
        train_data.loc[train_data["label"] == "noop", "label"] = 0

    return train_test_split(
        train_data.drop("label", axis=1).to_numpy(dtype=np.float32),
        train_data["label"].to_numpy(dtype=np.int32),
        test_size=0.2,
        random_state=42,
    )


def make_model(input_shape, num_classes):
    input_layer = layers.Input(input_shape)

    conv1 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)

    conv2 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)

    conv3 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)

    gap = layers.GlobalAveragePooling1D()(conv3)

    output_layer = layers.Dense(num_classes, activation="softmax")(gap)

    return keras.Model(inputs=input_layer, outputs=output_layer)


def train(short=False):
    # Load data from CSV files
    train_data = load_data(short=short)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = split_data(train_data, short=short)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    num_classes = len(np.unique(y_train))

    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    logical_device_count = len(tf.config.list_logical_devices("GPU"))

    timestamp = int(time())
    model_name = f"models/classifiers/{timestamp}_model" if not short else f"models/classifiers/{timestamp}_model_short"

    if logical_device_count > 0:
        with tf.device("/device:GPU:0"):
            model = make_model(input_shape=x_train.shape[1:], num_classes=num_classes)

        epochs = 200
        batch_size = 64

        callbacks_array = [
            callbacks.ModelCheckpoint(f"{model_name}.keras", save_best_only=True, monitor="val_loss"),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=0.0001),
            callbacks.EarlyStopping(monitor="val_loss", patience=35, verbose=1),
        ]
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_array,
            validation_split=0.2,
            verbose=1,
        )

        model = keras.models.load_model(model_name)

        test_loss, test_acc = model.evaluate(x_test, y_test)

        print("Test accuracy", test_acc)
        print("Test loss", test_loss)

        metric = "sparse_categorical_accuracy"
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.title("model " + metric)
        plt.ylabel(metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")
        plt.savefig(f"{model_name}_{metric}.png")
        plt.close()


def k_fold_crossvalidation_train():
    num_folds = 25

    # Load data from CSV files
    train_data = load_data()
    train_data.loc[train_data["label"] == "good", "label"] = 2
    train_data.loc[train_data["label"] == "noop", "label"] = 1
    train_data.loc[train_data["label"] == "bad", "label"] = 0

    inputs = train_data.drop(["label"], axis=1).to_numpy(dtype=np.float32)
    targets = train_data["label"].to_numpy(dtype=np.int32)

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
    num_classes = len(np.unique(targets))
    timestamp = int(time())

    # K-fold Cross Validation model evaluation
    fold_no = 1
    with tf.device("/device:GPU:0"):
        for train, test in kfold.split(inputs, targets):
            shape = (54, 1)
            model = make_model(input_shape=shape, num_classes=num_classes)

            callbacks_array = [
                callbacks.ModelCheckpoint(
                    f"models/classifiers/{timestamp}_model_1min.keras",
                    save_best_only=True,
                    monitor="sparse_categorical_accuracy",
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="sparse_categorical_accuracy", factor=0.5, patience=8, min_lr=0.0001
                ),
                callbacks.EarlyStopping(monitor="sparse_categorical_accuracy", patience=32, verbose=1),
            ]
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["sparse_categorical_accuracy"],
            )

            # Generate a print
            print("------------------------------------------------------------------------")
            print(f"Training for fold {fold_no} ...")

            # Fit data to model
            model.fit(
                inputs[train],
                targets[train],
                batch_size=64,
                epochs=50,
                verbose=1,
                callbacks=callbacks_array,
            )

            # Generate generalization metrics
            scores = model.evaluate(inputs[test], targets[test], verbose=0)
            print(
                f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%"
            )
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            # Increase fold number
            fold_no = fold_no + 1

    # == Provide average scores ==
    print("------------------------------------------------------------------------")
    print("Score per fold")
    for i in range(0, len(acc_per_fold)):
        print("------------------------------------------------------------------------")
        print(f"> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%")
    print("------------------------------------------------------------------------")
    print("Average scores for all folds:")
    print(f"> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})")
    print(f"> Loss: {np.mean(loss_per_fold)}")
    print("------------------------------------------------------------------------")


if __name__ == "__main__":
    train()
