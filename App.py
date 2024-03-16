import io
import cv2
import time
import requests
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Print version information
st.write("OpenCV ==", cv2.__version__)
st.write("Streamlit ==", st.__version__)
st.write("Pillow ==", Image.__version__)
st.write("numpy ==", np.__version__)
st.write("tensorflow ==", tf.__version__)
st.write("requests ==", requests.__version__)
# Add more print statements for other libraries/modules as needed

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
    linkedin_icon_path = r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\linkedin.jpg"
    email_icon_path = r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\email.png"

    # Create columns to display images and information for the first row
    col1, col2, col3 = st.columns([1, 0.1, 1])

    # Load and display the images using PIL for the first row
    with col1:
        person1_image = Image.open(r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\Mohammadjavad Hamidia.jpg")
        st.image(person1_image, use_column_width=True)
        st.header("Mohammadjavad Hamidia")
        st.markdown("### Assistant Professor")
        st.write("Department of Civil, Water and Environmental Engineering at Shahid Beheshti University")
        
        ############################### Email
        # Fetch the image from the URL
        response = requests.get("https://cdn-icons-png.freepik.com/256/552/552486.png")
        image_bytes = io.BytesIO(response.content)

        # Open the image using PIL
        email_icon = Image.open(image_bytes)

        # Resize the image
        email_icon_resized = email_icon.resize((30, 30))

        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(email_icon_resized)
        col2.write("[Email](mailto:m_hamidia@sbu.ac.ir)")

        ############################### LinkedIn
        # Fetch the image from the URL
        response = requests.get("https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png")
        image_bytes = io.BytesIO(response.content)

        # Open the image using PIL
        linkedin_icon = Image.open(image_bytes)

        # Resize the image
        linkedin_icon_resized = linkedin_icon.resize((30, 30))

        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(linkedin_icon_resized)
        col2.write("[LinkedIn](https://www.linkedin.com/in/mohammadjavadhamidia)")

    with col3:
        person2_image = Image.open(r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\Samane Rezaei.jpg")
        st.image(person2_image, use_column_width=True)
        st.header("Samane Rezaei")
        st.markdown("### Ph.D. Student in Structural Engineering")
        st.write("Department of Civil Engineering at Sharif University of Technology")

        ############################### Email
        # Fetch the image from the URL
        response = requests.get("https://cdn-icons-png.freepik.com/256/552/552486.png")
        image_bytes = io.BytesIO(response.content)

        # Open the image using PIL
        email_icon = Image.open(image_bytes)

        # Resize the image
        email_icon_resized = email_icon.resize((30, 30))

        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(email_icon_resized)
        col2.write("[Email](samane.rezaei@sharif.edu)")

        ############################### LinkedIn
        # Fetch the image from the URL
        response = requests.get("https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png")
        image_bytes = io.BytesIO(response.content)

        # Open the image using PIL
        linkedin_icon = Image.open(image_bytes)

        # Resize the image
        linkedin_icon_resized = linkedin_icon.resize((30, 30))

        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(linkedin_icon_resized)
        col2.write("[LinkedIn](https://www.linkedin.com/in/samane-rezaei-3999a5212)")

    # Create columns to display the image and information for the second row
    col4, col5, col6 = st.columns([1, 0.1, 1])

    with col4:
        person3_image = Image.open(r"C:\Users\samanerezaei\Desktop\Github\Concrete-Column-Heat-Maps\Kiarash Dolatshahi.jpeg")
        st.image(person3_image, use_column_width=True)
        st.header("Kiarash M.Dolatshahi")
        st.markdown("### Assistant Professor")
        st.write("Department of Civil Engineering at Sharif University of Technology")

        ############################### Email
        # Fetch the image from the URL
        response = requests.get("https://cdn-icons-png.freepik.com/256/552/552486.png")
        image_bytes = io.BytesIO(response.content)

        # Open the image using PIL
        email_icon = Image.open(image_bytes)

        # Resize the image
        email_icon_resized = email_icon.resize((30, 30))

        # Display the email icon with defined width and height
        col1, col2 = st.columns([0.2, 1])
        col1.image(email_icon_resized)
        col2.write("[Email](dolatshahi@sharif.edu)")

        ############################### LinkedIn
        # Fetch the image from the URL
        response = requests.get("https://cdn1.iconfinder.com/data/icons/logotypes/32/circle-linkedin-512.png")
        image_bytes = io.BytesIO(response.content)

        # Open the image using PIL
        linkedin_icon = Image.open(image_bytes)

        # Resize the image
        linkedin_icon_resized = linkedin_icon.resize((30, 30))

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
        my_bar = st.progress(0)
        
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
            
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