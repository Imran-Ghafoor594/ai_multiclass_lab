from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)

from src.config import (
    RF_ESTIMATORS,
    RF_MAX_DEPTH,
    AB_ESTIMATORS,
    RANDOM_STATE
)


def train_random_forest(X_train, y_train):

    rf_model = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    rf_model.fit(X_train, y_train)

    return rf_model


def train_adaboost(X_train, y_train):

    ab_model = AdaBoostClassifier(
        n_estimators=AB_ESTIMATORS,
        random_state=RANDOM_STATE
    )

    ab_model.fit(X_train, y_train)

    return ab_model