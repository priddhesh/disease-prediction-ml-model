from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def predictDisease(symptoms1):
    DATA_PATH = "./Training_disease_Prediction.csv"
    data = pd.read_csv(DATA_PATH).dropna(axis=1)

    disease_counts = data["prognosis"].value_counts()
    temp_df = pd.DataFrame({
        "Disease": disease_counts.index,
        "Counts": disease_counts.values
    })

    encoder = LabelEncoder()
    data["prognosis"] = encoder.fit_transform(data["prognosis"])

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24)

    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")

    def cv_scoring(estimator, X, y):
        return accuracy_score(y, estimator.predict(X))

    models = {
        "SVC": SVC(),
        "Gaussian NB": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=18)
    }

    # Producing cross validation score for the models
    for model_name in models:
        model = models[model_name]
        scores = cross_val_score(model, X, y, cv=10,
                                 n_jobs=-1,
                                 scoring=cv_scoring)

    svm_model = SVC()
    svm_model.fit(X_train, y_train)


    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(random_state=18)
    rf_model.fit(X_train, y_train)

    final_svm_model = SVC()
    final_nb_model = GaussianNB()
    final_rf_model = RandomForestClassifier(random_state=18)
    final_svm_model.fit(X, y)
    final_nb_model.fit(X, y)
    final_rf_model.fit(X, y)

    test_data = pd.read_csv(
        "./Testing_diesease_Prediction.csv").dropna(axis=1)

    test_X = test_data.iloc[:, :-1]
    test_Y = encoder.transform(test_data.iloc[:, -1])

    svm_preds = final_svm_model.predict(test_X)
    nb_preds = final_nb_model.predict(test_X)
    rf_preds = final_rf_model.predict(test_X)

    from statistics import mode

    symptoms = X.columns.values

    symptom_index = {}
    for index, value in enumerate(symptoms):
        symptom = " ".join([i.capitalize() for i in value.split("_")])
        symptom_index[symptom] = index
        print(value)

    data_dict = {
        "symptom_index": symptom_index,
        "predictions_classes": encoder.classes_
    }

    symptoms = symptoms1.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
    return final_prediction

app = Flask(__name__)


 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data['symptoms']
    prediction = predictDisease(symptoms)
    return jsonify({"prediction": prediction})
    
if __name__ == '__main__':
    app.run(debug=True)

