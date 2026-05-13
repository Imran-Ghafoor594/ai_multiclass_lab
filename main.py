import mlflow
import mlflow.sklearn

from src.preprocessing import (
    load_data,
    preprocess_data
)

from src.train import (
    train_random_forest,
    train_adaboost
)

from src.evaluate import evaluate_model

from src.utils import save_model


df = load_data()

X_train, X_test, y_train, y_test = preprocess_data(df)


with mlflow.start_run():

    # Random Forest

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    accuracy, precision, recall, f1 = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    mlflow.log_param("model", "RandomForest")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        rf_model,
        "random_forest_model"
    )

    save_model(
        rf_model,
        "models/random_forest.pkl"
    )

    # AdaBoost

    ab_model = train_adaboost(
        X_train,
        y_train
    )

    accuracy, precision, recall, f1 = evaluate_model(
        ab_model,
        X_test,
        y_test
    )

    mlflow.log_param("model_2", "AdaBoost")

    mlflow.log_metric("ab_accuracy", accuracy)
    mlflow.log_metric("ab_precision", precision)
    mlflow.log_metric("ab_recall", recall)
    mlflow.log_metric("ab_f1_score", f1)

    mlflow.sklearn.log_model(
        ab_model,
        "adaboost_model"
    )

    save_model(
        ab_model,
        "models/adaboost.pkl"
    )

print("\nPipeline executed successfully")