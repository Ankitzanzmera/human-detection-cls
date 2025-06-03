from PIL import Image
import streamlit as st

from human_detection_cls.prediction_st import PredictionPipeline

def predict_image(image):
    pipeline = PredictionPipeline()
    return pipeline.pred(image)

st.title("Binary Classification...")
uploaded_file = st.file_uploader("Upload Image For Prediction : ", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
        
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Predict"):
        prob, prediction, grad_cam_image = predict_image(image)
        st.markdown(f"### Prediction: {prediction} (Confidence: {prob:.2f})")
        
        with col2:
            st.image(grad_cam_image, caption="Grad-CAM Output", use_container_width=True)