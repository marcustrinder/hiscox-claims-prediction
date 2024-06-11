import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

def train_model(df: pd.DataFrame):
    X, y = df.drop('claim_status', axis=1), df[['claim_status']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1889)
    eval_set = [(X_train, y_train)]
    eval_metrics = ['auc', 'rmse', 'logloss']
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric=eval_metrics, enable_categorical=True)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=10)
    joblib.dump(model, 'models/xgboost_model.joblib')
    return model, X_test, y_test

def optimize_model(X_train, y_train):
    param_grid = {
        'n_estimators': stats.randint(50, 500),
        'learning_rate': stats.uniform(0.01, 0.75),
        'subsample': stats.uniform(0.25, 0.75),
        'max_depth': stats.randint(1, 8),
        'colsample_bytree': stats.uniform(0.1, 0.75),
        'min_child_weight': [1, 3, 5, 7, 9],
    }
    grid_search = RandomizedSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', enable_categorical=True), param_distributions=param_grid, cv=5, n_iter=100, scoring='roc_auc', verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
