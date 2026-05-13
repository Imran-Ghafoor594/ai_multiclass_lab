from sklearn.ensemble import AdaBoostClassifier

from src.preprocessing import (
    load_data,
    preprocess_data
)

from src.evaluate import evaluate_model


df = load_data()

X_train, X_test, y_train, y_test = preprocess_data(df)


ab_model = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

ab_model.fit(X_train, y_train)

evaluate_model(
    ab_model,
    X_test,
    y_test
)