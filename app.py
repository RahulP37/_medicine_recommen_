from flask import Flask, render_template, request
import os
import pickle
import numpy as np
import pandas as pd
import re
from lime.lime_tabular import LimeTabularExplainer
import speech_recognition as sr
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment

# ---------------------
# Initialize Flask app
# ---------------------
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac'}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
# Remove old LIME files
for f in os.listdir(STATIC_DIR):
    if f.startswith('lime_') and f.endswith('.html'):
        os.remove(os.path.join(STATIC_DIR, f))

# ---------------------
# Load Models & Data
# ---------------------
with open(os.path.join(BASE_DIR, 'models', 'symp_dict.pkl'), 'rb') as f:
    symp_dict = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'disease_dict.pkl'), 'rb') as f:
    disease_dict = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'svc.pkl'), 'rb') as f:
    svc = pickle.load(f)
with open(os.path.join(BASE_DIR, 'models', 'encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

precautions_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'precautions_df.csv'))
description_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'description.csv'))
medications_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'medications.csv'))
diets_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'diets.csv'))
workout_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'workout_df.csv'))

feature_names = list(symp_dict.keys())
explainer = LimeTabularExplainer(
    training_data=np.identity(len(feature_names)),
    feature_names=feature_names,
    class_names=list(label_encoder.classes_),
    discretize_continuous=False
)

# ---------------------
# Utility Functions
# ---------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = [word for word in text.split() if word not in {'the', 'is', 'and', 'in', 'to', 'of', 'a'}]
    return tokens

def extract_symptoms_bert(text, symp_dict, model, threshold=0.7):
    vocab = list(symp_dict.keys())
    vocab_phrases = [s.replace('_', ' ') for s in vocab]
    symp_emb = model.encode(vocab_phrases)

    tokens = preprocess_text(text)
    candidates = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)] + tokens
    cand_emb = model.encode(candidates)

    found = set()
    for i, c in enumerate(cand_emb):
        sims = cosine_similarity([c], symp_emb)[0]
        max_idx = np.argmax(sims)
        if sims[max_idx] >= threshold:
            found.add(vocab[max_idx])
    return list(found)

def transcribe_audio(filepath):
    recognizer = sr.Recognizer()

    # No need for mp3 conversion
    with sr.AudioFile(filepath) as source:
        audio = recognizer.record(source)

    try:
        return recognizer.recognize_google(audio)
    except Exception:
        return ''

def get_top_predictions(symptoms, topk=3):
    x = np.zeros(len(feature_names))
    for s in symptoms:
        idx = symp_dict.get(s)
        if idx is not None:
            x[idx] = 1
    if hasattr(svc, 'decision_function'):
        scores = svc.decision_function([x])[0]
    else:
        scores = svc.predict_proba([x])[0]
    top_idxs = np.argsort(scores)[-topk:][::-1]
    return [disease_dict[i] for i in top_idxs], x

def helper(disease):
    desc = ' '.join(description_df[description_df['Disease'] == disease]['Description'].fillna('').values)
    pre = precautions_df[precautions_df['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.tolist()
    med = medications_df[medications_df['Disease'] == disease]['Medication'].dropna().tolist()
    diet = diets_df[diets_df['Disease'] == disease]['Diet'].dropna().tolist()
    work = workout_df[workout_df['disease'] == disease]['workout'].dropna().tolist()
    return desc, [p for row in pre for p in row if pd.notna(p)], med, diet, work

# ---------------------
# Flask Routes
# ---------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check for text input first
        user_text = request.form.get('user_text', '').strip()

        if user_text:
            transcript = user_text
        else:
            # If no text, fall back to file
            file = request.files.get('file')
            if not file or not allowed_file(file.filename):
                return render_template('index.html', error="Upload a valid audio file (.wav, .mp3, .flac) or enter text.")
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            transcript = transcribe_audio(filepath)
            if not transcript:
                return render_template('index.html', error="Could not transcribe audio. Try again.")

        # Extract symptoms from transcript
        symptoms = extract_symptoms_bert(transcript, symp_dict, model)
        if not symptoms:
            return render_template('index.html', error="No known symptoms were detected.")

        print("Transcript:", transcript)
        print("Symptoms:", symptoms)

        predictions, input_vector = get_top_predictions(symptoms)

        results = []
        for disease in predictions:
            desc, pre, med, diet, work = helper(disease)
            exp = explainer.explain_instance(input_vector, svc.predict_proba, num_features=10)
            lime_path = os.path.join(STATIC_DIR, f'lime_{disease}.html')
            exp.save_to_file(lime_path)

            results.append({
                'disease': disease,
                'description': desc,
                'precautions': pre,
                'medications': med,
                'diets': diet,
                'workouts': work,
                'lime_html': f'lime_{disease}.html'
            })

        return render_template(
            'result.html',
            results=results,
            transcript=transcript,
            symptoms=symptoms
        )

    return render_template('index.html')



# ---------------------
# Run Server
# ---------------------
if __name__ == '__main__':
    app.run(debug=True)
