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

    st.write("""
    <p style='text-align: justify;'>This paper provides a probabilistic framework for quantifying the spatial distribution of cracking and crushing in rectangular reinforced concrete columns. 
                                    The probabilistic spatial analysis is accomplished on a rectangular reinforced concrete columns database tested under quasi-static cyclic loading. The database 
                                    includes 422 images of 109 damaged rectangular reinforced concrete columns with various geometry and material properties at different drift ratios between 0.2% and 6.0%. 
                                    Damaged heat maps derived from the probabilistic spatial analysis show the concentration and severity of the damage by highlighting the column areas that are more prone 
                                    to cracking and crushing. According to the three major categories for the aspect ratio of concrete columns, this study presented the damage heat maps in different ranges 
                                    of drift ratios and strength-based damage index (DIs) for each category. In the following, a set of classification models are generated based on the aspect ratio, crack 
                                    and crush indices of damaged rectangular reinforced concrete columns to predict the heat map level. The predicted heat map level shows the range of the drift ratio and 
                                    strength-based damage index (DIs) that each concrete column experienced. The results of this study are a useful benchmark for the reconnaissance teams to accelerate the 
                                    detection of the current status of the damaged concrete columns after an earthquake.</p>
    """, unsafe_allow_html=True)

    st.markdown('### Authors:')
    # Define paths to social media icons
    
    ##### Email
    response_email = requests.get("https://cdn-icons-png.freepik.com/256/552/552486.png")
    image_bytes_email = io.BytesIO(response_email.content)
    email_icon = Image.open(image_bytes_email)
    email_icon_resized = email_icon.resize((30, 30))
    
    ###### Linkedin
    response_linkedin = requests.get("https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png")
    image_bytes_linkedin = io.BytesIO(response_linkedin.content)
    linkedin_icon = Image.open(image_bytes_linkedin)
    linkedin_icon_resized = linkedin_icon.resize((30, 30))
        
    # Create columns to display images and information for the first row
    col1, col2, col3 = st.columns([1, 0.1, 1])

    # Load and display the images using PIL for the first row
    with col1:
        # Fetch the image from the URL
        image_url = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Images/Mohammadjavad%20Hamidia.jpg?raw=true"
        response = requests.get(image_url)

        # Open the image using PIL
        person1_image = Image.open(io.BytesIO(response.content))
        st.image(person1_image, use_column_width=True)
        st.header("Mohammadjavad Hamidia")
        st.markdown("### Assistant Professor")
        st.write("Department of Civil, Water and Environmental Engineering at Shahid Beheshti University")
        
        ############################### Email
        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(email_icon_resized)
        col2.write("[Email](mailto:m_hamidia@sbu.ac.ir)")

        ############################### LinkedIn
        # Fetch the image from the URL
        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(linkedin_icon_resized)
        col2.write("[LinkedIn](https://www.linkedin.com/in/mohammadjavadhamidia)")

    with col3:
        # Fetch the image from the URL
        image_url = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Images/Samane%20Rezaei.jpg?raw=true"
        response = requests.get(image_url)

        # Open the image using PIL
        person2_image = Image.open(io.BytesIO(response.content))
        st.image(person2_image, use_column_width=True)
        st.header("Samane Rezaei")
        st.markdown("### Ph.D. Student in Structural Engineering")
        st.write("Department of Civil Engineering at Sharif University of Technology")

        ############################### Email
        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(email_icon_resized)
        col2.write("[Email](samane.rezaei@sharif.edu)")

        ############################### LinkedIn
        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(linkedin_icon_resized)
        col2.write("[LinkedIn](https://www.linkedin.com/in/samane-rezaei-3999a5212)")

    # Create columns to display the image and information for the second row
    col4, col5, col6 = st.columns([1, 0.1, 1])

    with col4:
        # Fetch the image from the URL
        image_url = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Images/Kiarash%20Dolatshahi.jpeg?raw=true"
        response = requests.get(image_url)

        # Open the image using PIL
        person3_image = Image.open(io.BytesIO(response.content))

        # Display the image
        st.image(person3_image, use_column_width=True)
        st.header("Kiarash M.Dolatshahi")
        st.markdown("### Assistant Professor")
        st.write("Department of Civil Engineering at Sharif University of Technology")

        ############################### Email
        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(email_icon_resized)
        col2.write("[Email](dolatshahi@sharif.edu)")

        ############################### LinkedIn
        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(linkedin_icon_resized)
        col2.write("[LinkedIn](https://www.linkedin.com/in/kiarash-dolatshahi-0a20aba0)")

elif section == 'Guidelines':  

    st.markdown('### For using prediction models:')
    st.markdown('### Choose the type of heatmap')
    st.write("""
        <p style='text-align: justify;'>- This application is defined based on damage heat maps in different ranges of drift ratios and strength-based damage index (DIs) for three aspect ratio classes. For each type of heatmap, a classification model is provided that can be used to accelerate post-earthquake assessment. In this section, the user must choose the type of the heatmap (based on Drift, or DIS). </p>
        """, unsafe_allow_html=True)

    st.markdown('### Uploading Images')
    #st.write('- Click on the "Upload Images" button on the homepage.')
    st.write('- Select the images of the damaged concrete columns from your device.')
    st.write('- Ensure that the images are clear and show the damages accurately.')

    st.markdown('### Inputting Aspect Ratio Values')
    st.write('- Measure the dimensions of the concrete column (width, height, etc.).')
    st.write('- Calculate the aspect ratio value that is equal to dividing height by width.')
    st.write('- Input the aspect ratio value into the corresponding field on the application.')

    st.markdown('### Prediction')
    st.write('- Click on the "Predict" button to initiate the prediction process.')
    st.write('- Wait for the prediction models to process the inputs and generate results.')

    st.markdown('### Reviewing Results')
    st.write('- Once the assessment is complete, the results will be displayed on the screen.')
    st.write("""
        <p style='text-align: justify;'>- Related to the predicted class number of the heatmap, the range of the experienced drift or Lost resistance (based on the heatmap type) is determined, and the digitized image of the damaged concrete column (the image that shows the damaged zone of the column) will be shown as results. </p>
        """, unsafe_allow_html=True)


    st.markdown('### Additional Tips')
    st.write("""
        <p style='text-align: justify;'> - Ensure that the images provided are of high quality and clearly show the damages to get accurate predictions.</p>
        """, unsafe_allow_html=True)
             
    st.write("""
        <p style='text-align: justify;'> - Double-check the aspect ratio value inputted to ensure accuracy in the assessment.</p>
        """, unsafe_allow_html=True)

    st.write("""
        <p style='text-align: justify;'> - By following this guideline, you can effectively utilize the Post-Earthquake Concrete Column Assessment Tool to assess damaged concrete columns with confidence and make informed decisions regarding their condition and safety in post-earthquake scenarios. </p>
        """, unsafe_allow_html=True)

    st.write("""
        <p style='text-align: justify;'> - Please ensure that all the aforementioned guidelines are followed accurately to accurately predict the class of heatmaps associated with the uploaded image. Adherence to these guidelines is crucial for effectively assessing damaged concrete columns post-earthquake and ensuring accurate results..</p>
        """, unsafe_allow_html=True)


 
elif section == 'Prediction':
    # Function to preprocess the image for model input
    def preprocess_image(image, target_size=(100, 100)):
        # Resize image to target size
        resized_image = image.resize(target_size)
        
        # Enhance contrast using PIL's ImageEnhance module
        enhancer = ImageEnhance.Contrast(resized_image)
        contrast_enhanced = enhancer.enhance(10.0)  # Adjust the enhancement factor as needed
        
        # Convert to grayscale
        gray = contrast_enhanced.convert('L')
        
        # Apply adaptive thresholding for better feature capture
        thresholded_image = gray.point(lambda p: p > 127 and 255)
        
        # Convert grayscale to RGB
        rgb_image = thresholded_image.convert('RGB')
        
        # Expand dimensions to match model input shape
        return np.expand_dims(np.array(rgb_image), axis=0)

    TYPE = st.selectbox('Select the type of Heat Map', ["Based on Drift", "Based on DIS"])

    # Load the model
    # Define model links based on the selected type
    if TYPE == "Based on Drift":
        EDP = 'Drift'
        meta_model_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/raw/main/Models%20of%20Drift%20classification/meta_model.h5"
        model1_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/raw/main/Models%20of%20Drift%20classification/model1.h5"
        model2_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/raw/main/Models%20of%20Drift%20classification/model2.h5"
    elif TYPE == "Based on DIS":
        EDP = 'DIS'
        meta_model_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/raw/main/Models%20of%20DIS%20classification/meta_model.h5"
        model1_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/raw/main/Models%20of%20DIS%20classification/model1.h5"
        model2_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/raw/main/Models%20of%20DIS%20classification/model2.h5"

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
    aspect = st.number_input('Enter the aspect ration (length to width ratio) of the column', min_value=0.0, value=0.0, step=0.1)

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
        my_bar = st.progress(0)
        
        for percent_complete in range(100):
            time.sleep(0.1)
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
        crack_resized = crack_image.resize((100, (int(aspect))*100))
        crushing_resized = crushing_image.resize((100, (int(aspect))*100))
        original_image_resized = original_image.resize((100, (int(aspect))*100))
        
        # Display the images with captions and increased font size
        col1.image(original_image_resized, caption='Original Image', use_column_width=True)
        col2.image(cv2.resize(preprocessed_img.squeeze(), (100, (int(aspect))*100)), caption='Digitized Image',use_column_width=True)
        col3.image(crack_resized, caption='Critical zones for Cracking Damege', use_column_width=True)
        col4.image(crushing_resized, caption='Critical zones for Crushing Damage', use_column_width=True)