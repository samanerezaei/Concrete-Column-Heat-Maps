import cv2
import time
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

section = st.sidebar.radio('Navigation', ['Guidelines','Home','Abstract','Dataset Introduction','Preprocessing','Prediction'])
## Visualization

if section == 'Guidelines':  
    #st.header('Application Guidelines')

    st.markdown('### Application Guidelines')
    st.write('- **Navigation: Use the sidebar to navigate through different sections of the application.**')

    st.markdown('### Home Page')
    st.write('- **Welcome Message: Introduction and purpose of the application.**')
    st.write('- **Developer Information: Contact and professional links.**')

    # Making the second title bold using Markdown formatting
    st.markdown('### Preprocessing')
    st.write('- **Preprocessing Steps: Description of data cleaning, imputation, and normalization.**')
    st.write('- **Model Performance Comparison: Interactive line plots for different preprocessing methods.**')

    st.markdown('### Prediction')
    st.write('- **User Input: Input sliders for various parameters.**')
    st.write('- **Prediction Trigger: Button to initiate prediction.**')
    st.write('- **Comparison Feature: Option to compare different preprocessing methods.**')
    
elif section == 'Home':
    st.header('Welcome to the:')
    
    st.markdown('### Concrete Compressive Strength Predictor')
    st.write("This application is designed to provide insights and predictions about the compressive strength of concrete mixtures containing silica Fume. The predictions are based on an ensemble of machine learning models optimized with the Gray Wolf Optimization algorithm, as detailed in our research paper 'Utilizing Ensemble Machine Learning and Gray Wolf Optimization to Predict the Compressive Strength of Silica Fume Mixtures.'")
    
    st.write('Concrete is a critical material in the construction industry, and its strength is a key factor in determining the longevity and safety of structures. Silica Fume, a byproduct of silicon and ferrosilicon alloy production, is often used in concrete to enhance its properties. Our application leverages advanced machine learning techniques to predict the compressive strength of concrete containing silica Fume, helping engineers and researchers make informed decisions in their construction projects.')
    st.write('Navigate through the app using the sidebar to explore more about the dataset, preprocessing steps, and to use the prediction tool.')
    
    # Create columns to display images and information
    col1, space1, col2, space2, col3 = st.columns([1, 0.1, 0.1, 0.1, 1])
    
    # Load and display the images using PIL
    with col1:
        person1_image = Image.open(r"C:\Users\samanerezaei\Desktop\Github\Mohammadjavad Hamidia.jpg")
        st.image(person1_image, use_column_width=True)
        st.header("Mohammadjavad Hamidia")
        st.write("Assistant Professor")
        st.write("Department of Civil, Water and Environmental Engineering at Shahid Beheshti University")

    with col3:
        person2_image = Image.open(r"C:\Users\samanerezaei\Desktop\Github\Samane Rezaei.jpeg")
        st.image(person2_image, use_column_width=True)
        st.header("Samane Rezaei")
        st.write("Ph.D. Student in Structural Engineering at Sharif University of Technology.")
        
    col4, space3, col5, space4, col6 = st.columns([1, 0.1, 0.1, 0.1, 1])
    
    with col4:
        person3_image = Image.open(r"C:\Users\samanerezaei\Desktop\Github\Kiarash Dolatshahi.jpeg")
        st.image(person3_image, use_column_width=True)
        st.header("Kiarash M.Dolatshahi")
        st.write("Assistant Professor")
        st.write("Department of Civil Engineering at Sharif University of Technology")

        
#elif section == 'Abstract':
    
    
#elif section == 'Dataset Introduction':
    
    
elif section == 'Prediction':
    # Function to preprocess the image for model input
    def preprocess_image(image, target_size=(100, 100)):
        # Resize image to target size
        resized_image = cv2.resize(image, target_size)
        # Convert to grayscale
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast using histogram equalization
        enhanced_image = cv2.equalizeHist(gray)
        
        # Apply adaptive thresholding for better feature capture
        _, thresholded_image = cv2.threshold(enhanced_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert grayscale to RGB
        rgb_image = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2RGB)
        
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
    aspect = st.number_input('Enter the aspect ration (length to width ratio) of the column', min_value=0.0, value=0.0, step=0.1)

    if aspect <= 2:
        AspectCategory = '0 - 2'
    elif (aspect > 2) and (aspect <= 4):
        AspectCategory = '2 - 4'
    elif aspect > 4:
        AspectCategory = '4 - 100'
        
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
        predicted_class = predicted_label[0] + 1

        # Load the model
        if TYPE == "Based on Drift":
            if predicted_class == 1:
                Range = '0.0 t0 0.5'
            elif predicted_class == 2:
                Range = '0.5 t0 1.0'
            elif predicted_class == 3:
                Range = '1.0 t0 1.5'
            elif predicted_class == 4:
                Range = '1.5 t0 2.0'
            elif predicted_class == 5:
                Range = '2.0 t0 2.5'
            elif predicted_class == 6:
                Range = '2.5 t0 3.0'
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
        #my_bar = st.progress(0)
        
        #for percent_complete in range(100):
        #    time.sleep(0.1)
        #    my_bar.progress(percent_complete + 1)
            
        # Display the predicted class with increased font size
        st.markdown(f"<h3>Predicted Class: {predicted_class}</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3>{EDP_text}</h3>", unsafe_allow_html=True)
            
        # Construct file paths for Crack and Crushing images
        Crack = f'E:/sharif/Papers/Heat map/Concrete/Final Heat Maps/{TYPE}/Aspect Ratio {AspectCategory}/Crack/{predicted_class}.jpeg'
        Crushing = f'E:/sharif/Papers/Heat map/Concrete/Final Heat Maps/{TYPE}/Aspect Ratio {AspectCategory}/Crushing/{predicted_class}.jpeg'
        
        # Create columns to display images in a line
        col1, col2, col3, col4 = st.columns(4)
        
    # Load the images using PIL
        crack_image = Image.open(Crack)
        crushing_image = Image.open(Crushing)
        original_image = Image.open(uploaded_image)
        crack_resized = crack_image.resize((100, (int(aspect))*100))
        crushing_resized = crushing_image.resize((100, (int(aspect))*100))
        original_image_resized = original_image.resize((100, (int(aspect))*100))
        
        # Display the images with captions and increased font size
        col1.image(original_image_resized, caption='Original Image', use_column_width=True)
        col2.image(cv2.resize(preprocessed_img.squeeze(), (100, (int(aspect))*100)), caption='Digitized Image',use_column_width=True)
        col3.image(crack_resized, caption='Critical zones for Cracking Damege', use_column_width=True)
        col4.image(crushing_resized, caption='Critical zones for Crushing Damage', use_column_width=True)