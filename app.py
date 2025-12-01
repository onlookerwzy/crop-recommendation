# app.py
import streamlit as st
import numpy as np
import torch
import joblib
from kan import KAN # The KAN class is needed to reconstruct the model

# --- Configuration and Model Loading ---
st.set_page_config(page_title="Intelligent Crop Recommender", page_icon="🌾", layout="wide")

# Use caching to load model and helpers only once, improving performance
@st.cache_resource
def load_kan_model():
    """Loads the KAN model from the .pth file."""
    model = KAN(width=[7, 32, 16, 22], grid=3, k=3)
    # Load the saved state dictionary. map_location ensures it works on CPU-only deployment environments.
    model.load_state_dict(torch.load('kan_crop_model.pth', map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    return model

@st.cache_data
def load_helpers():
    """Loads the scaler and label encoder."""
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return scaler, label_encoder

# Load all necessary components
try:
    kan_model = load_kan_model()
    scaler, le = load_helpers()
except Exception as e:
    st.error(f"Error loading model or helper files: {e}")
    st.stop()

# --- Streamlit Web Interface ---
st.title("🌾 Intelligent Crop Recommendation System")
st.markdown("Leveraging a cutting-edge **Kolmogorov-Arnold Network (KAN)** to provide precise crop recommendations based on your soil and environmental conditions.")

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Enter Your Farm's Data")
    with st.form("input_form"):
        n = st.slider('Nitrogen (N) Content (kg/ha)', 0, 140, 90, help="Amount of Nitrogen in the soil.")
        p = st.slider('Phosphorous (P) Content (kg/ha)', 5, 145, 42, help="Amount of Phosphorous in the soil.")
        k = st.slider('Potassium (K) Content (kg/ha)', 5, 205, 43, help="Amount of Potassium in the soil.")
        temp = st.slider('Temperature (°C)', 8.0, 44.0, 25.6, step=0.1, help="Average temperature in your area.")
        humidity = st.slider('Relative Humidity (%)', 14.0, 100.0, 71.4, step=0.1, help="Average relative humidity.")
        ph = st.slider('Soil pH Value', 3.5, 9.9, 6.5, step=0.1, help="pH value of the soil.")
        rainfall = st.slider('Rainfall (mm)', 20.0, 299.0, 103.4, step=0.1, help="Average rainfall in mm.")
        
        submitted = st.form_submit_button("🌱 Recommend Crop")

with col2:
    st.subheader("Recommendation Result")
    if submitted:
        # 1. Create a numpy array from user input
        input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        
        # 2. Scale the input data using the loaded scaler
        scaled_input = scaler.transform(input_data)
        
        # 3. Convert to a PyTorch tensor
        input_tensor = torch.from_numpy(scaled_input).float()
        
        # 4. Make prediction
        with torch.no_grad():
            output = kan_model(input_tensor)
            _, predicted_idx_tensor = torch.max(output, 1)
            predicted_idx = predicted_idx_tensor.item()
        
        # 5. Decode the prediction to a crop name
        predicted_crop = le.inverse_transform([predicted_idx])[0]

        st.success(f"**Based on the provided data, the most suitable crop to cultivate is:**")
        st.markdown(f"<h1 style='text-align: center; color: #2ca02c;'>{predicted_crop.title()}</h1>", unsafe_allow_html=True)
    else:
        st.info("Please fill in the data on the left and click the button to get your recommendation.")

st.markdown("---")
st.write("Project by: [Your Name/Team Name] | Powered by Streamlit and PyTorch (KAN)")