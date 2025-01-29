import os
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import cv2

model = load_model('model_save/recyclable_classifier_model.h5')
classes = sorted(os.listdir(r'C:\Users\Harshan\Downloads\main-main\dataset\images\images'))

non_recyclable_classes = {"adapter", "aerosal_cans", "aluminium_food_cans", "mouse", "pen"}
recyclable_classes = {"office_paper", "paper_cups", "waste_paper", "cardboard_boxes", "cardboard_packaging", "leaves"}

def preprocess_image(image):
    image = cv2.resize(image, (200, 200))
    img_array = image / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    if predicted_class in recyclable_classes:
        recyclable_status = "Recyclable"
    else:
        recyclable_status = "Non-Recyclable"

    return predicted_class, confidence, recyclable_status

def main():
    st.title("Waste Classification App")
    option = st.selectbox("Choose Input Method", ["Upload an Image", "Live Camera"])

    if option == "Upload an Image":
        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            predicted_class, confidence, recyclable_status = predict(image)
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.write(f"Status: **{recyclable_status}**")

    elif option == "Live Camera":
        st.write("Capture an image for classification")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            st.image(frame, caption="Captured Image", channels="BGR", use_column_width=True)

            predicted_class, confidence, recyclable_status = predict(frame)
            st.write(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.write(f"Status: **{recyclable_status}**")
        else:
            st.write("Failed to capture image from camera")

if __name__ == "__main__":
    main()
