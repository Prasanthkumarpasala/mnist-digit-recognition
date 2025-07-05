from flask import Flask, request, jsonify
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import os
import numpy as np

app = Flask(__name__)

model_path = "saved_model/model.pkl"

@app.route('/training', methods=['POST'])
def train_model():
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

    rf = RandomForestClassifier()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='hard')

    with mlflow.start_run():
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", accuracy)

        os.makedirs("saved_model", exist_ok=True)
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, "ensemble_model")

    return jsonify({"message": "Model trained", "accuracy": accuracy})

@app.route('/prediction', methods=['POST'])
def predict():
    input_data = request.get_json()
    features = np.array(input_data["features"]).reshape(1, -1)

    if not os.path.exists(model_path):
        return jsonify({"error": "Train the model first."}), 400

    model = joblib.load(model_path)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

@app.route('/best_model_parameter', methods=['GET'])
def best_model_param():
    return jsonify({"note": "Model uses VotingClassifier with XGBoost and RandomForest. Details logged in MLflow."})

if __name__ == '__main__':
    app.run(debug=True)
