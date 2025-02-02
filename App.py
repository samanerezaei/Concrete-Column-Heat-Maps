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

@st.cache
def load_models(model1_link, model2_link, meta_model_link):
    try:
        # Download and save model files temporarily
        response_meta_model = requests.get(meta_model_link, stream=True)
        response_model1 = requests.get(model1_link, stream=True)
        response_model2 = requests.get(model2_link, stream=True)

        with open("meta_model.h5", "wb") as f:
            f.write(response_meta_model.content)
        with open("model1.h5", "wb") as f:
            f.write(response_model1.content)
        with open("model2.h5", "wb") as f:
            f.write(response_model2.content)

        # Load the models
        meta_model = load_model("meta_model.h5")
        model1 = load_model("model1.h5")
        model2 = load_model("model2.h5")
        return meta_model, model1, model2

    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        raise


# Preprocessing function
import cv2
import numpy as np
from PIL import Image

def enhance_image_contrast(image):
    """ Apply CLAHE for local contrast enhancement """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def detect_cracks(image, column_mask):
    """ Detect thin cracks using Canny edge detection with enhanced contrast """
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)
    cracks = cv2.bitwise_and(edges, column_mask)  # Mask cracks within the column
    return cracks

def detect_crushing(image, column_mask):
    """ Detect crushing areas using adaptive thresholding and morphological analysis """
    # Adaptive Thresholding for dark regions
    crushing = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8
    )
    crushing = cv2.bitwise_and(crushing, column_mask)

    # Remove small noise and keep only large connected components
    kernel = np.ones((7, 7), np.uint8)
    crushing_cleaned = cv2.morphologyEx(crushing, cv2.MORPH_CLOSE, kernel)
    
    # Connected Components to isolate large regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crushing_cleaned, connectivity=8)
    crushing_mask = np.zeros_like(image)
    for i in range(1, num_labels):  # Ignore the background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 500:  # Threshold for large regions
            crushing_mask[labels == i] = 255
    return crushing_mask

def process_damaged_image(image, target_size=(224, 224)):
    """
    Process the image to:
    - Detect column boundary
    - Highlight safe zones
    - Detect cracks and crushing areas
    - Generate final digitized output
    """
    # Step 1: Convert to grayscale
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np

    # Step 2: Column boundary detection
    _, thresholded = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    column_mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(column_mask, [largest_contour], -1, 255, thickness=-1)

    # Step 3: Enhance contrast
    contrast_enhanced = enhance_image_contrast(gray)

    # Step 4: Detect cracks and crushing separately
    cracks_mask = detect_cracks(contrast_enhanced, column_mask)
    crushing_mask = detect_crushing(contrast_enhanced, column_mask)

    # Step 5: Combine results
    final_output = np.full_like(gray, 255)  # White background
    final_output[crushing_mask > 0] = 0  # Crushing areas as solid black
    final_output[cracks_mask > 0] = 0  # Cracks as thin black lines

    # Resize to the target size
    final_resized = cv2.resize(final_output, target_size)
    return final_resized

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
email_icon = Image.open(io.BytesIO(response_email.content)).resize((30, 30))

response_linkedin = requests.get("https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png")
linkedin_icon = Image.open(io.BytesIO(response_linkedin.content)).resize((30, 30))

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
    }
]

# Create two columns
col1, col2 = st.columns(2)

# Display person details
for idx, author in enumerate(authors):
    with [col1, col2][idx]:  # Assign each person to col1 or col2
        response = requests.get(author["image_url"])
        person_image = Image.open(io.BytesIO(response.content))
        
        st.image(person_image, use_column_width=True)
        st.header(author["name"])
        st.markdown(f"### {author['title']}")
        st.write(author["affiliation"])

# Create another row for email & LinkedIn (Avoiding nested columns)
col1, col2 = st.columns(2)

for idx, author in enumerate(authors):
    with [col1, col2][idx]:  
        col_email, col_linkedin = st.columns([1, 1])  # Each column gets equal width
        with col_email:
            st.image(email_icon, width=25)
            st.write(f"[Email]({author['email']})")
        with col_linkedin:
            st.image(linkedin_icon, width=25)
            st.write(f"[LinkedIn]({author['linkedin']})")

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
        st.markdown("### Professor")
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
        

    with col6:
        # Fetch the image from the URL
        image_url = "https://raw.githubusercontent.com/samanerezaei/Concrete-Column-Heat-Maps/refs/heads/main/Images/Asjodi.jpg"
        response = requests.get(image_url)

        # Open the image using PIL
        person4_image = Image.open(io.BytesIO(response.content))

        # Display the image
        st.image(person4_image, use_column_width=True)
        st.header("Amir Hossein Asjodi")
        st.markdown("### Assistant Professor")
        st.write("Department of Civil Engineering at K.N. Toosi University of Technology")

        ############################### Email
        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(email_icon_resized)
        col2.write("[Email](amir.asjodi74@gmail.com)")

        ############################### LinkedIn
        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(linkedin_icon_resized)
        col2.write("[LinkedIn](https://www.linkedin.com/in/amir-hossein-asjodi/)")
        
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
    TYPE = st.selectbox('Select the type of Heat Map', ["Based on Drift", "Based on DIS"])

    # Load the models ONCE
    if TYPE == "Based on Drift":
        EDP = 'Drift'
        meta_model_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Models%20of%20Drift%20classification/meta_model.h5"
        model1_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Models%20of%20Drift%20classification/model1.h5"
        model2_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Models%20of%20Drift%20classification/model2.h5"
        
    elif TYPE == "Based on DIS":
        EDP = 'DIS'
        meta_model_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Models%20of%20DIS%20classification/meta_model.h5"
        model1_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Models%20of%20DIS%20classification/model1.h5"
        model2_link = "https://github.com/samanerezaei/Concrete-Column-Heat-Maps/blob/main/Models%20of%20DIS%20classification/model2.h5"
    
    #st.write("Loading models... Please wait.")
    meta_model, model1, model2 = load_models(model1_link, model2_link, meta_model_link)

    # response_meta_model = requests.get(meta_model_link)
    # response_model1 = requests.get(model1_link)
    # response_model2 = requests.get(model2_link)

    # Save the downloaded content to temporary files
    # with open("meta_model.h5", "wb") as f:
    #     f.write(response_meta_model.content)
    # with open("model1.h5", "wb") as f:
    #     f.write(response_model1.content)
    # with open("model2.h5", "wb") as f:
    #     f.write(response_model2.content)


    # Load the meta-model and other models from temporary files
    # meta_model = load_model("meta_model.h5")
    # model1 = load_model("model1.h5")
    # model2 = load_model("model2.h5")

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
    uploaded_image = st.file_uploader('Upload an image of a damaged RC column', type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        # Load and display the original image
        img = Image.open(uploaded_image)
        #st.image(img, caption='Uploaded Image', use_column_width=True)
    
        # Process the image
        binary_img = process_damaged_image(img)
    
        # Convert single-channel binary image to three channels
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
    
        # Handle prediction results based on selected type
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
        col2.image(cv2.resize(binary_img.squeeze(), (100, (int(aspect))*100)), caption='Digitized Image',use_column_width=True)
        col3.image(crack_resized, caption='Critical zones for Cracking Damege', use_column_width=True)
        col4.image(crushing_resized, caption='Critical zones for Crushing Damage', use_column_width=True)