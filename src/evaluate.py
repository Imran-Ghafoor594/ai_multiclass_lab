from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average='weighted'
    )

    recall = recall_score(
        y_test,
        predictions,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        predictions,
        average='weighted'
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\nAccuracy:", accuracy)

    print("\nPrecision:", precision)

    print("\nRecall:", recall)

    print("\nF1 Score:", f1)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("\nConfusion Matrix:\n")

    print(cm)

    return accuracy, precision, recall, f1