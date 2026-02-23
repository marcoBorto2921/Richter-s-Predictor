# src/model_utils.py

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def train_xgboost(X_train, y_train, X_test, y_test):
    
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Cross validation (macro F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro"
    )

    print("CV Macro F1:", cv_scores.mean())
    
    # Fit finale
    model.fit(X_train, y_train,eval_set=[(X_test, y_test)],verbose=True)

    # Test evaluation
    y_pred = model.predict(X_test)

    print("\nTest Macro F1:", f1_score(y_test, y_pred, average="macro"))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return model