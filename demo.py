import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

@st.cache_resource
def load_models():
    heart_model = tf.keras.models.load_model("/home/ariakame/Documents/IDP/cnn_melspec_model.h5")
    lung_model = tf.keras.models.load_model("/home/ariakame/Documents/IDP/cnn_melspec_model.h5")
    return heart_model, lung_model

heart_model, lung_model = load_models()

heart_class_names = ['Abnormal', 'Normal']
lung_class_names = ['COPD', 'Asthma', 'Healthy']  # Edit as per your trained classes

# Preprocessing functions
def preprocess_heart_audio(file_path, sr=4000, n_mels=128, duration=4):
    y, _ = librosa.load(file_path, sr=sr)
    max_len = sr * duration
    y = y[:max_len] if len(y) > max_len else np.pad(y, (0, max_len - len(y)))
    mel_spec = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_db = np.expand_dims(mel_db, axis=(0, -1))  # Shape: (1, 128, time, 1)
    return mel_db

def preprocess_lung_audio(file_path, sr=22050, n_mels=128):
    y, _ = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_db = np.expand_dims(mel_db, axis=(0, -1))  # Shape: (1, 128, time, 1)
    return mel_db

# Streamlit UI
st.title("🩺 Heart & Pulmonary Disorder Classification")

model_choice = st.selectbox("Choose Model", ["Heart Disorder Classification", "Pulmonary Disorder Classification"])
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    try:
        if model_choice == "Heart Disorder Classification":
            processed = preprocess_heart_audio(uploaded_file)
            prediction = heart_model.predict(processed)
            predicted_class = int(np.round(prediction[0][0]))
            confidence = float(prediction[0][0])
            st.success(f"🩺 Predicted: **{heart_class_names[predicted_class]}** (Confidence: {confidence:.2f})")

        else:
            processed = preprocess_lung_audio(uploaded_file)
            prediction = lung_model.predict(processed)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            st.success(f"🫁 Predicted: **{lung_class_names[predicted_class]}** (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"❌ Error during prediction:\n\n{e}")
