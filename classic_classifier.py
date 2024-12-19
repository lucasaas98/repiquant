# Standard Library
from time import time

# Third party dependencies
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from joblib import dump
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# Current project dependencies
import data_api
import helper as h


def load_data(fast=False, short=False):
    all_tickers = data_api.get_actionable_stocks_list()
    interval = "5min"

    first = True
    all_data = None

    for ticker in all_tickers:
        if first:
            data = pl.scan_parquet(h.get_scaled_labeled_parquet(ticker, interval, short=short))
            all_data = all_data.select(
                pl.exclude(["avg_stoploss", "avg_bars_in_market", "avg_takeprofit", "avg_return_pct", "datetime"])
            ).collect()
            first = False
        else:
            data = pl.scan_parquet(h.get_scaled_labeled_parquet(ticker, interval, short=short))
            data = data.select(
                pl.exclude(["avg_stoploss", "avg_bars_in_market", "avg_takeprofit", "avg_return_pct", "datetime"])
            ).collect()
            all_data = all_data.merge_sorted(data)
        if fast:
            break

    return all_data


def train_classifier_simple():
    train_data = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        train_data.select(pl.exclude("label")).to_numpy(),
        train_data.select(pl.col("label")).to_numpy(),
        test_size=0.2,
        random_state=42,
    )

    svm_model = svm.SVC(kernel="rbf", gamma=0.5, C=1.0)
    svm_model.fit(X_train, y_train)

    dump(svm_model, f"{time()}_svc.model")

    y_pred = svm_model.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))
    print(metrics.balanced_accuracy_score(y_test, y_pred))
    cm = metrics.confusion_matrix(y_test, y_pred, labels=svm_model.classes_)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    print(metrics.classification_report(y_test, y_pred, target_names=svm_model.classes_))


def train_classifier():
    train_data = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        train_data.select(pl.exclude("label")).to_numpy(),
        train_data.select(pl.col("label")).to_numpy(),
        test_size=0.2,
        random_state=42,
    )

    pipe = Pipeline([("classifier", RandomForestClassifier())])

    param_grid = [
        # {
        #     "classifier": [LogisticRegression()],
        #     "classifier__penalty": ["l1", "l2"],
        #     "classifier__C": np.logspace(-4, 4, 20),
        #     "classifier__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        #     "classifier__max_iter": list(range(50, 500, 100)),
        # },
        {
            "classifier": [RandomForestClassifier()],
            "classifier__n_estimators": list(range(10, 101, 10)),
            "classifier__max_features": list(range(6, 32, 5)),
        },
        # {
        #     "classifier": [svm.SVC()],
        #     "classifier__kernel": ["linear", "rbf"],
        #     "classifier__gamma": np.logspace(-4, 4, 20),
        #     "classifier__C": np.logspace(-4, 4, 20),
        # },
    ]

    clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=4)

    best_clf = clf.fit(X_train, y_train)

    print("Best Parameters:", best_clf.best_params_)

    print("Model accuracy is", best_clf.score(X_test, y_test))
    probs = best_clf.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label="GridSearchCV (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig("Log_ROC.png")

    print("AUC is", roc_auc)

    classes = best_clf.predict(X_test)

    accuracy = metrics.accuracy_score(classes, y_test)

    balanced_accuracy = metrics.balanced_accuracy_score(classes, y_test)

    precision = metrics.precision_score(classes, y_test)

    average_precision = metrics.average_precision_score(classes, y_test)

    f1_score = metrics.f1_score(classes, y_test)

    recall = metrics.recall_score(classes, y_test)

    print(accuracy, balanced_accuracy, precision, average_precision, f1_score, recall, roc_auc)

    print(metrics.classification_report(classes, y_test))


if __name__ == "__main__":
    train_classifier_simple()
