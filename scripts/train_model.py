import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

def main(args):
    df = pd.read_csv(args.input_data)
    X, y = df.drop('claim_status', axis=1), df['claim_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1889)
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', enable_categorical=True)
    model.fit(X_train, y_train)
    joblib.dump(model, args.model_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='input.csv')
    parser.add_argument('--model-output', type=str, default='model.joblib')
    args = parser.parse_args()
    main(args)
