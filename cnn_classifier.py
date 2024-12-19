# Standard Library
from time import time

# Third party dependencies
import keras
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import callbacks, layers
from tensorflow.keras.optimizers import Adam

# Current project dependencies
import helper as h


def load_data(fast=False, short=False):
    all_tickers = h.get_reasonable_tickers()
    interval = "5min"

    first = True
    all_data = None

    for ticker in all_tickers:
        if first:
            data = pl.scan_parquet(h.get_scaled_labeled_parquet(ticker, interval, short=short))
            all_data = data.select(
                pl.exclude(["avg_stoploss", "avg_bars_in_market", "avg_takeprofit", "avg_return_pct"])
            ).collect()
            first = False
        else:
            data = pl.scan_parquet(h.get_scaled_labeled_parquet(ticker, interval, short=short))
            data = data.select(
                pl.exclude(["avg_stoploss", "avg_bars_in_market", "avg_takeprofit", "avg_return_pct"])
            ).collect()
            all_data = all_data.merge_sorted(data, key="datetime")
        if fast:
            break

    return all_data.select(pl.exclude("datetime"))


def load_data_both_sides(fast=False):
    all_tickers = h.get_reasonable_tickers()
    interval = "5min"

    first = True
    all_data = None

    for ticker in all_tickers:
        if first:
            data = pl.scan_parquet(h.get_scaled_labeled_both_sides_parquet(ticker, interval))
            all_data = data.select(
                pl.exclude(["avg_stoploss", "avg_bars_in_market", "avg_takeprofit", "avg_return_pct"])
            ).collect()
            first = False
        else:
            data = pl.scan_parquet(h.get_scaled_labeled_both_sides_parquet(ticker, interval))
            data = data.select(
                pl.exclude(["avg_stoploss", "avg_bars_in_market", "avg_takeprofit", "avg_return_pct"])
            ).collect()
            all_data = all_data.merge_sorted(data, key="datetime")
        if fast:
            break

    return all_data.select(pl.exclude("datetime"))


def split_data(train_data, short=False):
    train_data_df = None
    if not short:
        train_data_df = train_data.with_columns(
            label=pl.when(pl.col("label") == "very good")
            .then(2)
            .when(pl.col("label") == "noop")
            .then(1)
            .when(pl.col("label") == "very bad")
            .then(0)
            .otherwise(-1),
        )
    else:
        train_data_df = train_data.with_columns(
            label=pl.when(pl.col("label") == "very bad")
            .then(2)
            .when(pl.col("label") == "bad")
            .then(1)
            .when(pl.col("label") == "noop")
            .then(0)
            .otherwise(-1),
        )

    return train_test_split(
        train_data_df.select(pl.exclude("label")).to_numpy(),
        train_data_df.select(pl.col("label")).to_numpy(),
        test_size=0.2,
        random_state=42,
    )


def split_data_both_sides(train_data):
    train_data_df = train_data.with_columns(
        label=pl.when(pl.col("label") == "very good")
        .then(2)
        .when(pl.col("label") in ["noop", "good", "bad"])
        .then(1)
        .when(pl.col("label") == "very bad")
        .then(0)
        .otherwise(-1),
    )

    return train_test_split(
        train_data_df.select(pl.exclude("label")).to_numpy(),
        train_data_df.select(pl.col("label")).to_numpy(),
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
    train_data = load_data(short=short)

    x_train, x_test, y_train, y_test = split_data(train_data, short=short)

    unique, counts = np.unique(y_train, return_counts=True)
    num_classes = len(unique)
    print(f"(no resampling) Label: {unique}")
    print(f"(no resampling) Label Counts: {counts}")

    # cc = ClusterCentroids(random_state=42)
    # x_train_resampled, y_train_resampled = cc.fit_resample(x_train, y_train)

    smote = SMOTE(random_state=42, n_jobs=-1)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    x_train = x_train_resampled.reshape((x_train_resampled.shape[0], x_train_resampled.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    y_train = y_train_resampled

    unique, counts = np.unique(y_train_resampled, return_counts=True)
    num_classes = len(unique)
    print(f"(resampling) Label: {unique}")
    print(f"(resampling) Label Counts: {counts}")

    logical_device_count = len(tf.config.list_logical_devices("GPU"))

    timestamp = int(time())
    model_name = f"models/classifiers/{timestamp}_model" if not short else f"models/classifiers/{timestamp}_model_short"

    if logical_device_count > 0:
        with tf.device("/device:GPU:0"):
            model = make_model(input_shape=x_train.shape[1:], num_classes=num_classes)

        epochs = 400
        batch_size = 128

        callbacks_array = [
            callbacks.ModelCheckpoint(f"{model_name}.keras", save_best_only=True, monitor="val_loss"),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
            callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
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

        model = keras.models.load_model(f"{model_name}.keras")

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


def train_both_sides():
    train_data = load_data_both_sides()

    x_train, x_test, y_train, y_test = split_data_both_sides(train_data)

    unique, counts = np.unique(y_train, return_counts=True)
    num_classes = len(unique)
    print(f"(no resampling) Label: {unique}")
    print(f"(no resampling) Label Counts: {counts}")

    # cc = ClusterCentroids(random_state=42)
    # x_train_resampled, y_train_resampled = cc.fit_resample(x_train, y_train)

    smote = SMOTE(random_state=42, n_jobs=-1)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    x_train_resampled = x_train_resampled.reshape((x_train_resampled.shape[0], x_train_resampled.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    unique, counts = np.unique(y_train_resampled, return_counts=True)
    num_classes = len(unique)
    print(f"(resampling) Label: {unique}")
    print(f"(resampling) Label Counts: {counts}")

    logical_device_count = len(tf.config.list_logical_devices("GPU"))

    timestamp = int(time())
    model_name = f"models/classifiers/{timestamp}_model_both_sides"

    if logical_device_count > 0:
        with tf.device("/device:GPU:0"):
            model = make_model(input_shape=x_train_resampled.shape[1:], num_classes=num_classes)

        epochs = 400
        batch_size = 64

        callbacks_array = [
            callbacks.ModelCheckpoint(f"{model_name}.keras", save_best_only=True, monitor="val_loss"),
            # callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
            callbacks.LearningRateScheduler(schedule=lambda epoch, lr: lr * 0.95 ** (epoch // 5)),
            callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
        history = model.fit(
            x_train_resampled,
            y_train_resampled,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_array,
            validation_split=0.2,
            verbose=1,
        )

        model = keras.models.load_model(f"{model_name}.keras")

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

        y_pred = model.predict(x_test)
        y_pred_class = np.argmax(y_pred, axis=1)

        print("Accuracy:", accuracy_score(y_test, y_pred_class))
        print(classification_report(y_test, y_pred_class))
        print(confusion_matrix(y_test, y_pred_class))


def make_model_temporal(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)

    conv1 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)

    pool1 = layers.MaxPooling1D(pool_size=2)(conv1)

    conv2 = layers.Conv1D(filters=128, kernel_size=3, padding="same")(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)

    pool2 = layers.MaxPooling1D(pool_size=2)(conv2)

    gap = layers.GlobalAveragePooling1D()(pool2)

    output_layer = layers.Dense(num_classes, activation="sigmoid")(gap)

    return keras.Model(inputs=input_layer, outputs=output_layer)


def split_data_new_model(train_data):
    train_data_df = train_data.with_columns(
        label=pl.when(pl.col("label") in ["good", "very good"])
        .then(1)
        .when(pl.col("label") in ["noop", "very bad", "bad"])
        .then(0)
        .otherwise(-1),
    )

    return train_test_split(
        train_data_df.select(pl.exclude("label")).to_numpy(),
        train_data_df.select(pl.col("label")).to_numpy(),
        test_size=0.2,
        random_state=42,
    )


def create_binary_model(input_shape):
    model = keras.Sequential(
        [
            layers.Input(input_shape),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def train_new_model():
    train_data = load_data()

    x_train, x_test, y_train, y_test = split_data_new_model(train_data)

    unique, counts = np.unique(y_train, return_counts=True)

    print(f"(no resampling) Label: {unique}")
    print(f"(no resampling) Label Counts: {counts}")

    # cc = ClusterCentroids(random_state=42)
    # x_train_resampled, y_train_resampled = cc.fit_resample(x_train, y_train)

    smote = SMOTE(random_state=42, n_jobs=-1)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    # x_train_resampled = x_train_resampled.reshape((x_train_resampled.shape[0], x_train_resampled.shape[1], 1))
    # x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print(f"(resampling) Label: {unique}")
    print(f"(resampling) Label Counts: {counts}")

    logical_device_count = len(tf.config.list_logical_devices("GPU"))

    timestamp = int(time())
    model_name = f"models/classifiers/{timestamp}_model_both_sides"

    if logical_device_count > 0:
        with tf.device("/device:GPU:0"):
            model = create_binary_model(input_shape=x_train_resampled.shape[1:])

        epochs = 1000
        batch_size = 1024

        callbacks_array = [
            callbacks.ModelCheckpoint(f"{model_name}.keras", save_best_only=True, monitor="val_loss"),
            callbacks.LearningRateScheduler(
                schedule=lambda epoch, lr: lr * 0.95 if (epoch % 10) == 0 and epoch not in [0, 1, 2, 3, 4] else lr
            ),
            callbacks.EarlyStopping(monitor="val_loss", patience=100, verbose=1),
        ]
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.TruePositives(name="tp"),
                tf.keras.metrics.FalsePositives(name="fp"),
                tf.keras.metrics.TrueNegatives(name="tn"),
                tf.keras.metrics.FalseNegatives(name="fn"),
            ],
        )

        history = model.fit(
            x_train_resampled,
            y_train_resampled,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks_array,
            validation_split=0.2,
            verbose=1,
        )

        model = keras.models.load_model(f"{model_name}.keras")

        test_loss, test_acc, test_auc, test_tp, test_fp, test_tn, test_fn = model.evaluate(x_test, y_test)

        print("Test accuracy", test_acc)
        print("Test loss", test_loss)

        metric = "accuracy"
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric])
        plt.title("model " + metric)
        plt.ylabel(metric, fontsize="large")
        plt.xlabel("epoch", fontsize="large")
        plt.legend(["train", "val"], loc="best")
        plt.savefig(f"{model_name}_{metric}.png")
        plt.close()


if __name__ == "__main__":
    train_new_model()
