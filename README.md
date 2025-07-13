# 🧠 Personalized Medicine Recommendation System (Voice-Enabled & Explainable)

This project is an end-to-end machine learning system that predicts the top-3 most likely diseases from a patient's described symptoms—either typed or spoken—and provides personalized recommendations, including medications, diets, workouts, and precautions.

It uses real-time voice transcription, intelligent symptom extraction via Sentence-BERT, and interpretable predictions using LIME.

---

## 🚀 Features

- 🔍 Predicts top-3 most probable diseases from symptoms
- 🎙️ Accepts symptoms via voice (.wav audio files) or text input
- 🤖 Tuned SVC classifier achieving **100% precision, recall, and F1-score** across **41 disease classes** (test set: 984 samples)
- 🧬 BERT-based symptom extractor with **optimized cosine similarity threshold = 0.7**
- 🌐 Web interface built using Flask + HTML/CSS
- 📊 Interactive model explanations using **LIME HTML charts**
- 🧠 Personalized outputs:
  - 💊 Medications
  - 🥗 Diet recommendations
  - 🏋️ Workouts
  - 🧼 Precautions

---

## 📷 Demo

🔗 [Click here to watch the demo video on Google Drive](https://drive.google.com/file/d/17YO_1d2iDu9mkfs2iLmomZcEegm3Q0PL/view?usp=sharing)

> If it doesn’t preview directly in-browser, click the download icon in the top-right.

---

## 🛠 Tech Stack

- **Programming Language**: Python
- **ML & NLP**:
  - scikit-learn (SVC Classifier, GridSearchCV)
  - Sentence-BERT (symptom matching via cosine similarity)
  - LIME (model explainability)
- **Speech Processing**: `SpeechRecognition`, `PyDub`
- **Web**: Flask, HTML/CSS
- **Data Handling**: pandas, NumPy

---

## 📊 Model Performance

Trained a multi-class SVC model using symptom-based one-hot vectors. Performed hyperparameter tuning using GridSearchCV.

📈 **Classification Report (SVC - Test Set of 984 samples, 41 classes)**:
## 📊 Model Performance
<p align="center">
  <img src="https://raw.githubusercontent.com/RahulP37/_medicine_recommen_/main/assets/Screenshot%202025-07-13%20074727.png" width="700">
</p>






---

## 🧪 How It Works

1. User provides input via:
   - Typing symptoms directly
   - Uploading a `.wav` audio file of spoken symptoms
2. If audio: Transcription via Google Speech Recognition API
3. Symptoms are extracted using Sentence-BERT + cosine similarity
4. Disease prediction using trained SVC model
5. Top-3 predictions shown with:
   - Medication list
   - Diet suggestions
   - Recommended workouts
   - Precaution tips
6. LIME explainability shows symptom-level contributions via interactive HTML bar charts

---







## 🙋‍♂️ Acknowledgements

- [Sentence-BERT](https://www.sbert.net/)
- [scikit-learn](https://scikit-learn.org/)
- [LIME](https://github.com/marcotcr/lime)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)
