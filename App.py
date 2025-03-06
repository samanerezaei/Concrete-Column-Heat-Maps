import io
import os
import cv2
import time
import requests
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.mixture import GaussianMixture

# Fixing the download and loading of model files
@st.cache(allow_output_mutation=True)
def load_models(model1_link, model2_link, meta_model_link):
    try:
        # Define the local filenames
        local_model1 = "model1.h5"
        local_model2 = "model2.h5"
        local_meta_model = "meta_model.h5"
        
        # Function to download the model
        def download_model(url, filename):
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                return filename
            else:
                st.error(f"Failed to download the model from {url}. Status: {response.status_code}")
                return None

        # Download models
        model1_file = download_model(model1_link, local_model1)
        model2_file = download_model(model2_link, local_model2)
        meta_model_file = download_model(meta_model_link, local_meta_model)

        # If any model failed to download, return None
        if not all([model1_file, model2_file, meta_model_file]):
            return None, None, None
        
        # Load the models
        meta_model = load_model(meta_model_file)
        model1 = load_model(model1_file)
        model2 = load_model(model2_file)
        
        return meta_model, model1, model2

    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        return None, None, None

def enhance_image_contrast(image):
    """ Apply CLAHE for local contrast enhancement """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

import numpy as np
import cv2
from PIL import Image

def convert_pil_to_numpy(image):
    """Convert a PIL image to a NumPy array, ensuring grayscale format."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3:  # Convert RGB to grayscale if necessary
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    return image

def detect_cracks(image):
    """Detect cracks while preserving fine details. Cracks should be black (0) and background white (255)."""
    image = convert_pil_to_numpy(image)
    
    # Gaussian Blur to reduce noise
    #denoised = cv2.GaussianBlur(image, (3, 3), 0)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(15, 15))
    enhanced = clahe.apply(image)

    # Apply Canny Edge Detection to highlight crack edges
    edges = cv2.Canny(enhanced, 80, 180)

    # Ensure cracks are black and background is white
    cracks_output = np.full_like(edges, 255)
    cracks_output[edges > 0] = 0  # Turn edges black

    return cracks_output

def detect_crushing(image):
    """Detect crushing using Adaptive Thresholding and Connected Components."""
    image = convert_pil_to_numpy(image)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    # Adaptive Thresholding for primary crushing mask
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 50  # Reduced sensitivity
    )

    # Apply Morphological Closing to refine crushing regions
    kernel = np.ones((7, 7), np.uint8)
    crushing = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Remove small regions using Connected Components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crushing, connectivity=8)
    min_area = 100000  # Increase this threshold to avoid capturing shadowy areas
    filtered_crushing = np.zeros_like(crushing)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_crushing[labels == i] = 255

    # Ensure crushing is black and background is white
    crushing_output = np.full_like(filtered_crushing, 255)
    crushing_output[filtered_crushing > 0] = 0  # Set crushing areas to black

    return crushing_output

def process_damaged_image(image):
    """
    Process the image at high resolution first, then resize to (224, 224) with high quality.
    Detect cracks and crushing in separate masks and combine them using weighted addition.
    Ensure that crack areas are line-based (black) and crushing areas are area-based (black).
    """
    image = convert_pil_to_numpy(image)

    # Detect cracks and crushing at original size
    cracks_mask = detect_cracks(image)
    crushing_mask = detect_crushing(image)

    # Resize with better interpolation (INTER_CUBIC for higher quality)
    cracks_mask = cv2.resize(cracks_mask, (224, 224), interpolation=cv2.INTER_CUBIC)
    crushing_mask = cv2.resize(crushing_mask, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Now combine the masks using weighted sum for better control over combining
    combined_mask = cv2.addWeighted(cracks_mask, 1, crushing_mask, 0, 0)

    # Convert any non-white (255) pixels to black (0)
    combined_mask[combined_mask != 255] = 0

    return cracks_mask, crushing_mask, combined_mask

# Streamlit App Section
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
    <p style='text-align: justify;'>This paper provides a framework based on damage density heat maps to accelerate the post-earthquake assessment of rectangular reinforced concrete (RC) columns. 
    The heat maps utilize image processing filters to analyze visual characteristics of damaged RC columns, such as cracks and crushing areas. This visual analysis is accomplished on a database 
    of rectangular RC columns tested under quasi-static cyclic loading. The database includes 422 images of 109 damaged rectangular RC columns with various geometry (aspect ratios between 2 and 6.4) 
    and material properties at drift ratios between 0.2% and 6.0%. According to the database, three major categories exist for the aspect ratios of the RC columns. The damage density heat maps are 
    presented based on the ranges of the drift ratio and strength-based damage index (DIs) for each aspect ratio category. Indeed, the heat maps show the concentration and severity of the damage by 
    highlighting the column areas that are more prone to cracking and crushing.  In the following, two classification models utilizing machine learning methods are developed to predict the heatmap 
    level of damaged RC columns. The predicted heat map level determines the range of the experienced drift ratio and DIs (lost strength of the RC column) range. Finally, these models are made accessible 
    through a user-friendly web application, where input parameters such as the aspect ratio and images of damaged RC columns can be used to generate predictions. The results of this study are a useful 
    benchmark for the reconnaissance teams to accelerate the detection of the current status of the damaged concrete columns after an earthquake.</p>
    """, unsafe_allow_html=True)

    st.markdown('### Authors:')
    
    # Fetch icons
    response_email = requests.get("https://cdn-icons-png.freepik.com/256/552/552486.png")
    email_icon = Image.open(io.BytesIO(response_email.content)).resize((25, 25))
    
    response_linkedin = requests.get("https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png")
    linkedin_icon = Image.open(io.BytesIO(response_linkedin.content)).resize((25, 25))
    
    # Authors' data
    authors = [
        {
            "name": "Mohammadjavad Hamidia",
            "title": "Assistant Professor",
            "affiliation": "Department of Civil, Water and Environmental Engineering at Shahid Beheshti University",
            "image_url": "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Images/Mohammadjavad%20Hamidia.jpg?raw=true",
            "email": "mailto:m_hamidia@sbu.ac.ir",
            "linkedin": "https://www.linkedin.com/in/mohammadjavadhamidia"
        },
        {
            "name": "Samane Rezaei",
            "title": "Ph.D. Student in Structural Engineering",
            "affiliation": "Department of Civil Engineering at Sharif University of Technology",
            "image_url": "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Images/Samane%20Rezaei.jpg?raw=true",
            "email": "mailto:samane.rezaei@sharif.edu",
            "linkedin": "https://www.linkedin.com/in/samane-rezaei-3999a5212"
        },
        {
            "name": "Kiarash M.Dolatshahi",
            "title": "Professor",
            "affiliation": "Department of Civil Engineering at Sharif University of Technology",
            "image_url": "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Images/Kiarash%20Dolatshahi.jpeg?raw=true",
            "email": "mailto:dolatshahi@sharif.edu",
            "linkedin": "https://www.linkedin.com/in/kiarash-dolatshahi-0a20aba0"
        },
        {
            "name": "Amir Hossein Asjodi",
            "title": "Assistant Professor",
            "affiliation": "Department of Civil Engineering at K.N. Toosi University of Technology",
            "image_url": "https://raw.githubusercontent.com/samanerezaei/Concrete-Column-Heat-Maps/refs/heads/main/Images/Asjodi.jpg",
            "email": "mailto:amir.asjodi74@gmail.com",
            "linkedin": "https://www.linkedin.com/in/amir-hossein-asjodi/"
        }
    ]
    
    # Display authors in two rows, each containing two columns
    for row in range(0, len(authors), 2):
        col1, col2 = st.columns(2)
    
        for idx, col in enumerate([col1, col2]):
            if row + idx < len(authors):
                author = authors[row + idx]
    
                # Fetch and display image
                response = requests.get(author["image_url"])
                person_image = Image.open(io.BytesIO(response.content))
    
                with col:
                    st.image(person_image, use_column_width=True)
                    st.header(author["name"])
                    st.markdown(f"### {author['title']}")
                    st.write(author["affiliation"])
    
                    # Email & LinkedIn in the same row
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <a href="{author['email']}" style="text-decoration: none;">
                                <img src="https://cdn-icons-png.freepik.com/256/552/552486.png" width="25"> Email
                            </a>
                            <a href="{author['linkedin']}" style="text-decoration: none;">
                                <img src="https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png" width="25"> LinkedIn
                            </a>
                        </div>
                        """, unsafe_allow_html=True
                    )

        
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
    # Select the type of Heat Map
    TYPE = st.selectbox('Select the type of Heat Map', ["Based on Drift", "Based on DIS"])

    # Define model links based on the selected type
    if TYPE == "Based on Drift":
        EDP = 'Drift'
        meta_model_link = "https://raw.githubusercontent.com/samanerezaei/Concrete-Column-Heat-Maps/main/Models%20of%20Drift%20classification/meta_model.h5"
        model1_link = "https://raw.githubusercontent.com/samanerezaei/Concrete-Column-Heat-Maps/main/Models%20of%20Drift%20classification/model1.h5"
        model2_link = "https://raw.githubusercontent.com/samanerezaei/Concrete-Column-Heat-Maps/main/Models%20of%20Drift%20classification/model2.h5"
        
    elif TYPE == "Based on DIS":
        EDP = 'DIS'
        meta_model_link = "https://raw.githubusercontent.com/samanerezaei/Concrete-Column-Heat-Maps/main/Models%20of%20DIS%20classification/meta_model.h5"
        model1_link = "https://raw.githubusercontent.com/samanerezaei/Concrete-Column-Heat-Maps/main/Models%20of%20DIS%20classification/model1.h5"

    # Load models and test if they load successfully
    meta_model, model1, model2 = load_models(model1_link, model2_link, meta_model_link)

    # Test model file loading
    if meta_model and model1 and model2:
        st.success("Models loaded successfully!")
    else:
        st.error("Error loading models.")
        st.stop()  # Stop further execution if models can't be loaded

    # Get aspect ratio input from user
    aspect = st.number_input('Enter the aspect ratio (length to width ratio) of the column', min_value=0.0, value=0.0, step=0.1)

    if aspect <= 2:
        InitiateRange = '0'
        FinalRange = '2'
    elif (aspect > 2) and (aspect <= 4):
        InitiateRange = '2'
        FinalRange = '4'
    elif aspect > 4:
        InitiateRange = '4'
        FinalRange = '100'
        
    # Upload and process the image
    uploaded_image = st.file_uploader('Upload an image of a damaged RC column', type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
    
        # Process the image    
        # Process the image
        cracks_mask, crushing_mask, combined_mask = process_damaged_image(img)
        
        # Ensure processed images are in correct format for display
        cracks_mask_display = Image.fromarray(cracks_mask)
        crushing_mask_display = Image.fromarray(crushing_mask)
        combined_mask_display = Image.fromarray(combined_mask)

        # Display images separately
        # st.subheader("Detected Crack and Crushing Maps")
        # col1, col2, col3 = st.columns(3)
    
        # with col1:
        #     st.image(cracks_mask_display, caption="Crack Detection", use_column_width=True)
        
        # with col2:
        #     st.image(crushing_mask_display, caption="Crushing Detection", use_column_width=True)
            
        # with col3:
        #     st.image(combined_mask_display, caption="Crack + Crushing Detection", use_column_width=True)

        # Convert masks to binary (ensure they contain only 0 and 255)
        #cracks_mask = (cracks_mask > 0).astype(np.uint8) * 255
        #crushing_mask = (crushing_mask > 0).astype(np.uint8) * 255
        
        # Use bitwise OR to combine masks without losing information
        #binary_img = cv2.bitwise_or(cracks_mask, crushing_mask)
        binary_img = combined_mask
        
        if binary_img is None or binary_img.size == 0:
            raise ValueError("Error: The processed image is empty. Check the crack and crushing detection functions.")
        
        binary_img_display = Image.fromarray(combined_mask)
    
        # Display the final combined image
        #st.subheader("Final Combined Damage Map")
        #st.image(binary_img_display, caption="Final Damage Map (Cracks + Crushing)", use_column_width=True)
        
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)

    
        # Expand dimensions to match model input
        binary_img = np.expand_dims(binary_img, axis=0)  # Add batch dimension
    
        # Expand aspect to match batch dimension
        aspect_array = np.expand_dims([aspect], axis=0)
    
        # Make predictions using the loaded model
        predictions1 = model1.predict([binary_img, aspect_array])
        predictions2 = model2.predict(binary_img)
    
        # Combine predictions for the meta-model
        stacking_features = np.hstack((predictions1, predictions2))
        predicted_label = np.argmax(meta_model.predict(stacking_features), axis=1)
        predicted_class = predicted_label[0] + 1
    
        # Define result text based on prediction type
        if TYPE == "Based on Drift":
            drift_ranges = ['0.0 to 0.5', '0.5 to 1.0', '1.0 to 1.5', '1.5 to 2.0', '2.0 to 2.5', '2.5 to 3.0', 'more than 3.0']
            Range = drift_ranges[predicted_class - 1]
            EDP_text = f"The concrete column has experienced about {Range} percent drift ratio."
    
        elif TYPE == "Based on DIS":
            dis_ranges = ['20% to 35%', '35% to 50%', '50%', '50% to 65%', 'more than 65%']
            Range = dis_ranges[predicted_class - 1]
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
        
        # Fetch heat map images based on prediction
        crack_url = f'https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Final%20Heat%20Maps/Based%20on%20{EDP}/Aspect%20Ratio%20{InitiateRange}%20-%20{FinalRange}/Crack/{predicted_class}.jpeg?raw=true'
        crushing_url = f'https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Final%20Heat%20Maps/Based%20on%20{EDP}/Aspect%20Ratio%20{InitiateRange}%20-%20{FinalRange}/Crushing/{predicted_class}.jpeg?raw=true'
        response_crack = requests.get(crack_url)
        response_crushing = requests.get(crushing_url)
        
        # Display images in columns
        col1, col2, col3, col4 = st.columns(4)

        # Load and resize images
        crack_image = Image.open(io.BytesIO(response_crack.content))
        crushing_image = Image.open(io.BytesIO(response_crushing.content))
        original_image = Image.open(uploaded_image)
        
        img_height = int(aspect * 100)
        crack_resized = crack_image.resize((100, img_height))
        crushing_resized = crushing_image.resize((100, img_height))
        original_image_resized = original_image.resize((100, img_height))
        
        # Display images with captions
        col1.image(original_image_resized, caption='Original Image', use_column_width=True)
        col2.image(cv2.resize(binary_img.squeeze(), (100, img_height)), caption='Digitized Image', use_column_width=True)
        col3.image(crack_resized, caption='Critical zones for Cracking Damage', use_column_width=True)
        col4.image(crushing_resized, caption='Critical zones for Crushing Damage', use_column_width=True)