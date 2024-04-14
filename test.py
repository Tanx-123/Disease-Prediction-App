import gradio as gr
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset and model during initialization
model = load_model("disease_prediction_model.h5")
desc = pd.read_csv("symptom_Description.csv")
precautions = pd.read_csv("symptom_precaution.csv")

# Load the tokenizer and label encoder during initialization
with open("tokenizer.pickle", 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pickle', 'rb') as f:
    label_encoder = pickle.load(f)

def get_description_and_precautions(disease):
    disease_lower = disease.lower()
    if disease_lower in [d.lower() for d in desc["Disease"].values]:
        description = desc.loc[desc["Disease"].str.lower() == disease_lower, "Description"].values[0]
        precautions_list = precautions.loc[precautions["Disease"].str.lower() == disease_lower, "Precaution_1":"Precaution_4"].values
        if precautions_list.size > 0:
            precautions_list = precautions_list[0].tolist()
            precautions_list = [precaution for precaution in precautions_list if str(precaution) != 'nan']
            return description, precautions_list
        else:
            return description, ["No precautions found in the dataset."]
    else:
        return "Disease not found in the dataset.", []

def predict_disease(user_input):
    if not user_input.strip():
        return []
    user_input_seq = pad_sequences(tokenizer.texts_to_matrix([user_input],mode='tfidf'), maxlen=150, padding='post', truncating='post')
    predictions = model.predict(user_input_seq)[0]
    top_3_indices = predictions.argsort()[-3:][::-1]
    top_3_probabilities = [predictions[idx] for idx in top_3_indices]
    top_3_diseases = [label_encoder.inverse_transform([idx])[0] for idx in top_3_indices]
    top_3_descriptions = []
    top_3_precautions = []
    for disease in top_3_diseases:
        description, precautions = get_description_and_precautions(disease)
        top_3_descriptions.append(description)
        top_3_precautions.append(precautions)
    return list(zip(top_3_diseases, top_3_probabilities, top_3_descriptions, top_3_precautions))

def predict_and_display(user_input):
    top_3_predictions = predict_disease(user_input)
    output = ""
    if top_3_predictions:
        for disease, prob, description, precautions in top_3_predictions:
            output += f"{disease} ({prob*100:.2f}%)\n\n"
            output += f"Description: {description}\n\n"
            output += "Precautions:\n"
            for precaution in precautions:
                output += f"- {precaution}\n"
            output += "\n"
    else:
        output = "Please enter valid symptoms."
    return output

demo = gr.Interface(
    fn=predict_and_display,
    inputs=gr.Textbox(label="Enter Symptoms"),
    outputs=gr.Textbox(label="Prediction"),
    title="Disease Prediction",
    description="Enter your symptoms and get the top 3 disease predictions, along with descriptions and precautions.",
)

demo.launch(share=True)   