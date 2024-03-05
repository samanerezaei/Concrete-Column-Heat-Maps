import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess the image for model input
def preprocess_image(image):
    # Resize image to (100, 100)
    resized_image = cv2.resize(image, (100, 100))
    # Convert to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Convert grayscale image to RGB
    rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # Expand dimensions to match model input shape
    return np.expand_dims(rgb_image, axis=0)

TYPE = st.selectbox('Select the type of Heat Map', ["Based on Drift", "Based on DIS"])

# Load the model
if TYPE == "Based on Drift":  
    meta_model_path = r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\Models of Drift classification\meta_model.h5"
    model1_path = r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\Models of Drift classification\model1.h5"
    model2_path = r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\Models of Drift classification\model2.h5"

elif TYPE == "Based on DIS":  
    meta_model_path = r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\Models of DIS classification\meta_model.h5"
    model1_path = r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\Models of DIS classification\model1.h5"
    model2_path = r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\Models of DIS classification\model2.h5"

meta_model = load_model(meta_model_path)
model1 = load_model(model1_path)
model2 = load_model(model2_path)

# Get length and width values from the user
length = st.number_input('Enter the length of the column', min_value=0.0, value=0.0, step=0.1)
width = st.number_input('Enter the width of the column', min_value=0.0, value=0.0, step=0.1)

# Check if both length and width are provided
if length != 0.0 and width != 0.0:
    # Calculate aspect ratio
    aspect = length / width

    # Show length, width, and aspect ratio
    st.write(f"Length: {length}")
    st.write(f"Width: {width}")
    st.write(f"Aspect Ratio: {aspect}")
        
    # Load the image
    uploaded_image = st.file_uploader('Upload Image')
    if uploaded_image is not None:
        # Convert file uploader data to numpy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        # Decode image
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Preprocess the image
        preprocessed_img = preprocess_image(img)
        
        # Make predictions using the loaded model
        predictions1 = model1.predict(preprocessed_img)
        predictions2 = model2.predict([preprocessed_img, np.array([aspect])])
        stacking_features = np.hstack((predictions1, predictions2))
        predicted_label = np.argmax(meta_model.predict(stacking_features), axis=1)
        
        # Display the segmented areas
        st.image(predictions1, caption='Segmented Areas', use_column_width=True)
        
        # Display the predicted class
        st.write("Predicted Class:", predicted_label)
else:
    st.warning("Please enter both length and width values.")
