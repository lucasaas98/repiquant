# Third party dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

# Current project dependencies
import helper as h


def train_classifier():
    # Define the list of tickers to use
    all_tickers = ["TSLA", "AAPL", "MSFT", "GOOG", "AMZN"]
    interval = "5min"

    all_data = []

    # Load dataset from CSV files
    for ticker in all_tickers:
        scaled_labeled_50_bars = pd.read_csv(h.get_scaled_labeled(ticker, interval), index_col=0)

        scaled_labeled_50_bars = scaled_labeled_50_bars.drop(
            ["avg_stoploss", "avg_bars_in_market", "avg_takeprofit"], axis=1
        )
        all_data.append(scaled_labeled_50_bars)

    # Concatenate all dataframes into one dataframe
    train_data = pd.concat(all_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        train_data.drop("label", axis=1), train_data["label"], test_size=0.2, random_state=42
    )

    pipe = Pipeline([("classifier", RandomForestClassifier())])

    param_grid = [
        {
            "classifier": [LogisticRegression()],
            "classifier__penalty": ["l1", "l2"],
            "classifier__C": np.logspace(-4, 4, 20),
            "classifier__solver": ["liblinear"],
            "classifier__max_iter": list(range(50, 500, 100)),
        },
        {
            "classifier": [RandomForestClassifier()],
            "classifier__n_estimators": list(range(10, 101, 10)),
            "classifier__max_features": list(range(6, 32, 5)),
        },
    ]
    clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

    best_clf = clf.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best Parameters:", best_clf.best_params_)

    # # Train a logistic regression classifier based on the data and respective labels
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)

    # # Make predictions for the testing set
    # y_pred = best_clf.predict(X_test)

    # # Evaluate the model's accuracy
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

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

    return


if __name__ == "__main__":
    train_classifier()

    # Best Parameters: {'classifier': RandomForestClassifier(), 'classifier__max_features': 6, 'classifier__n_estimators': 70}
    # Model accuracy is 0.871661887038054
