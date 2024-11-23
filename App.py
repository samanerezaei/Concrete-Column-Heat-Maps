import io
import os
import cv2
import time
import requests
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

def preprocess_image(image, target_size=(224, 224)):
    # Resize image to target size (same as used for training models)
    resized_image = image.resize(target_size)

    # Enhance contrast using PIL's ImageEnhance module
    enhancer = ImageEnhance.Contrast(resized_image)
    contrast_enhanced = enhancer.enhance(2.0)  # Adjust contrast enhancement factor as needed

    # Convert image to RGB (if it's not already in RGB format)
    rgb_image = contrast_enhanced.convert('RGB')

    # Normalize image to match the input format used during training
    normalized_image = np.array(rgb_image) / 255.0

    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    return np.expand_dims(normalized_image, axis=0)

section = st.sidebar.radio('Navigation', ['Home','Guidelines','Prediction'])
    
## Visualization
if section == 'Home':
    st.header('Welcome to the:')
    
    st.markdown('### Damage Density Heatmaps Predictor')

    st.write("""
    <p style='text-align: justify;'>This application is designed to accelerate post-earthquake assessment. The predicted class of the heatmaps provides insight into the experienced drift 
                                    and lost strength of the concrete column. The predictions are based on an ensemble of machine learning models based on classification algorithms, as 
                                    detailed in our research paper 'Damage density heat maps of rectangular reinforced concrete columns.'
</p>
    """, unsafe_allow_html=True)

elif section == 'Prediction':
    TYPE = st.selectbox('Select the type of Heat Map', ["Based on Drift", "Based on DIS"])

    # Load the model
    # Define model links based on the selected type
    if TYPE == "Based on Drift":
        EDP = 'Drift'
        meta_model_link = "E:\sharif\Papers\Heat map\Python Code\Drift\Selected Classification\meta_best_model_20241115_004034.keras"
        model1_link = "E:\sharif\Papers\Heat map\Python Code\Drift\Selected Classification\best_model_1_20241114_223909.keras"
        model2_link = "E:\sharif\Papers\Heat map\Python Code\Drift\Selected Classification\final_base_model_2.h5"
    elif TYPE == "Based on DIS":
        EDP = 'DIS'
        meta_model_link = "E:\sharif\Papers\Heat map\Python Code\DIS\Classification 2\meta_best_model.h5"
        model1_link = "E:\sharif\Papers\Heat map\Python Code\DIS\Classification 2\final_base_model_1.h5"
        model2_link = "E:\sharif\Papers\Heat map\Python Code\DIS\Classification 2\final_base_model_2.h5"

    # Download the meta-model file
    response_meta_model = requests.get(meta_model_link)
    response_model1 = requests.get(model1_link)
    response_model2 = requests.get(model2_link)

    # Save the downloaded content to temporary files
    with open("meta_model.h5", "wb") as f:
        f.write(response_meta_model.content)
    with open("model1.h5", "wb") as f:
        f.write(response_model1.content)
    with open("model2.h5", "wb") as f:
        f.write(response_model2.content)

    # Load the meta-model and other models from temporary files
    meta_model = load_model("meta_model.h5")
    model1 = load_model("model1.h5")
    model2 = load_model("model2.h5")

    # Get length and width values from the user
    aspect = st.number_input('Enter the aspect ratio (length to width ratio) of the column', min_value=0.0, value=0.0, step=0.1)

    # Define aspect ratio categories based on user input
    if aspect <= 2:
        InitiateRange = '0'
        FinalRange = '2'
    elif (aspect > 2) and (aspect <= 4):
        InitiateRange = '2'
        FinalRange = '4'
    elif aspect > 4:
        InitiateRange = '4'
        FinalRange = '100'
        
    # Load the image
    uploaded_image = st.file_uploader('Upload Image')
    if uploaded_image is not None:
        # Convert file uploader data to numpy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        # Decode image
        img = Image.open(io.BytesIO(file_bytes))
        
        # Preprocess the image
        preprocessed_img = preprocess_image(img)
        
        # Make predictions using the loaded model
        predictions1 = model1.predict(preprocessed_img)
        predictions2 = model2.predict([preprocessed_img, np.array([aspect])])
        stacking_features = np.hstack((predictions1, predictions2))
        predicted_label = np.argmax(meta_model.predict(stacking_features), axis=1)
        predicted_class = predicted_label[0] + 1

        # Determine the predicted range based on the selected type
        if TYPE == "Based on Drift":
            if predicted_class == 1:
                Range = '0.0 to 0.5'
            elif predicted_class == 2:
                Range = '0.5 to 1.0'
            elif predicted_class == 3:
                Range = '1.0 to 1.5'
            elif predicted_class == 4:
                Range = '1.5 to 2.0'
            elif predicted_class == 5:
                Range = '2.0 to 2.5'
            elif predicted_class == 6:
                Range = '2.5 to 3.0'
            elif predicted_class == 7:
                Range = 'more than 3.0'
            
            EDP_text = f"The concrete column has experienced about {Range} percent drift ratio."
                
        elif TYPE == "Based on DIS":
            if predicted_class == 1:
                Range = '20% to 35%'
            elif predicted_class == 2:
                Range = '35% to 50%'
            elif predicted_class == 3:
                Range = '50%'
            elif predicted_class == 4:
                Range = '50% to 65%'
            elif predicted_class == 5:
                Range = 'more than 65%'
            
            EDP_text = f"The concrete column has lost {Range} of its strength."
           
    # Button to trigger prediction
    if st.button('Predict'):
        my_bar = st.progress(0)
        
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)
            
        # Display the predicted class with increased font size
        st.markdown(f"<h3>Predicted Class: {predicted_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3>{EDP_text}</h3>", unsafe_allow_html=True)
        
        # Fetch the image from the URL
        crack_url = f'https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Final%20Heat%20Maps/Based%20on%20{EDP}/Aspect%20Ratio%20{InitiateRange}%20-%20{FinalRange}/Crack/{predicted_class}.jpeg?raw=true'
        crushing_url = f'https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Final%20Heat%20Maps/Based%20on%20{EDP}/Aspect%20Ratio%20{InitiateRange}%20-%20{FinalRange}/Crushing/{predicted_class}.jpeg?raw=true'
        response_crack = requests.get(crack_url)
        response_crushing = requests.get(crushing_url)
        
        # Create columns to display images in a line
        col1, col2, col3, col4 = st.columns(4)

        # Load the images using PIL
        crack_image = Image.open(io.BytesIO(response_crack.content))
        crushing_image = Image.open(io.BytesIO(response_crushing.content))
        original_image = Image.open(uploaded_image)
        crack_resized = crack_image.resize((224, int(aspect) * 224))
        crushing_resized = crushing_image.resize((224, int(aspect) * 224))
        original_image_resized = original_image.resize((224, int(aspect) * 224))
        
        # Display the images with captions and increased font size
        col1.image(original_image_resized, caption='Original Image', use_column_width=True)
        col2.image(cv2.resize(preprocessed_img.squeeze(), (224, int(aspect) * 224)), caption='Digitized Image', use_column_width=True)
        col3.image(crack_resized, caption='Critical zones for Cracking Damage', use_column_width=True)
        col4.image(crushing_resized, caption='Critical zones for Crushing Damage', use_column_width=True)
