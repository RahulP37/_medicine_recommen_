# utils.py
import numpy as np
import pandas as pd
import pickle
import os

# Load models
svc = pickle.load(open("models/svc.pkl", "rb"))
symp_dict = pickle.load(open("models/symp_dict.pkl", "rb"))
disease_dict = pickle.load(open("models/disease_dict.pkl", "rb"))

# Load datasets
description = pd.read_csv("data/description.csv")
precautions = pd.read_csv("data/precautions_df.csv")
medications = pd.read_csv("data/medications.csv")
diets = pd.read_csv("data/diets.csv")
workout = pd.read_csv("data/workout_df.csv")

def helper(dis):
    desc_series = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc_series.values)

    pre_df = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre_df.values.tolist()

    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].values.tolist()

    return desc, pre, med, die, wrkout

def get_predicted_value(patient_symptoms, model=svc):
    # Validate input symptoms
    for item in patient_symptoms:
        if item not in symp_dict:
            print(f"Unknown symptom: {item}")
            return "Invalid symptoms entered"

    # Create input vector
    input_vector = np.zeros(len(symp_dict))
    for item in patient_symptoms:
        input_vector[symp_dict[item]] = 1

    # Predict top 3 diseases
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([input_vector])[0]
    else:
        # Fallback if using SVC without predict_proba
        if hasattr(model, "decision_function"):
            probs = model.decision_function([input_vector])[0]
            # Normalize to act like probabilities
            probs = (probs - np.min(probs)) / (np.max(probs) - np.min(probs))
        else:
            # Fallback to single prediction
            return disease_dict[model.predict([input_vector])[0]]

    # Get top 3 indices
    top_indices = np.argsort(probs)[::-1][:3]
    top_diseases = [disease_dict[i] for i in top_indices]

    return top_diseases
