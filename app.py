from tensorflow.keras.models import load_model 
import plotly.express as px 
from PIL import Image
import streamlit as st 
import pickle
import numpy as np 
import pandas as pd 
import os 
import cv2
 
model_path = r'skin_classifier_model.h5'
model = load_model(model_path)
scaler_path = r'scaler.pkl'
scaler = pickle.load(open(scaler_path, 'rb'))
icon = '⚕️'
features = ['Age', 'Region', 'Itch', 'Grew', 'Pain', 'Changed', 'Bleed', 'Elevation']
diagnosis = {
    'MEL':'Melanoma', 'SCC':'Squamous Cell Carcinoma', 
    'BCC':'Basal Cell Carcinoma', 'NEV':'Nevus', 'ACK':'Actinic Keratosis', 
    'SEK': 'Seborrheic Keratosis'}
regions = {
    'ARM': 0, 'NECK': 1, 'FACE': 2, 'HAND': 3, 'FOREARM': 4, 'CHEST': 5, 'NOSE': 6, 
    'THIGH': 7, 'SCALP': 8, 'EAR': 9, 'BACK': 10, 'FOOT': 11, 'ABDOMEN': 12, 'LIP': 13
    }
maps = {'True':2, 'False':1, 'UNK':0}

colab_link = 'https://colab.research.google.com/drive/11P8rgP5yh2V8Hw7TuM1USnOBdo71Y4e6?usp=sharing'
report_link = 'https://docs.google.com/document/d/16R3mtugIjT-p4kP9Kd-87zqvyMCMKtYeW8R6UdNzS_Y/edit?usp=sharing'

def image_preprocessor(image): 
    size = (256, 256)
    img_arr = np.array(Image.open(image))
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    img_arr = cv2.resize(img_arr, size)
    img_arr = np.array(img_arr)/255
    
    return img_arr

def convert_dataframe(input_x2): 
    features = ['Standardized Age', 'Region', 'Itch', 'Grew', 'Pain', 'Changed', 'Bleed', 'Elevation']
    dataframe = pd.DataFrame(data = input_x2, columns = features)
    
    return dataframe

def make_prediction(model, input_x1, input_x2): 
    to_maps = input_x2[0,:]
    x2 = []

    for i in range(len(to_maps)): 
        val = to_maps[i]
        if i == 0: 
            x2.append(float(val))
        elif i == 1: 
            x2.append(float(regions[val]))
        else: 
            x2.append(float(maps[val]))

    x1 = input_x1
    x2 = np.array([x2])

    pred_proba = model.predict(x = [x1,x2])
    pred_idx = np.argmax(pred_proba, axis = 1)[0]
    diags = ['NEV', 'BCC', 'ACK', 'SEK', 'SCC', 'MEL']
    prediction = diags[pred_idx]

    return pred_proba, prediction 

def user_input(): 
    region_choice = [region for region in regions]
    map_choice = [map for map in maps]

    st.sidebar.subheader('Input Features')
    image = st.sidebar.file_uploader('Upload the image over here:', type = ['png', 'jpg'])
    st.sidebar.markdown('*Please ensure that the image covers only the skin and the lesion, if cannot, it is advisable to crop out the background.*')
    age = st.sidebar.slider('How old is the patient?', 5.0, 120.0, 30.0)
    region = st.sidebar.selectbox('Where is the lesion found?', region_choice)
    itch = st.sidebar.selectbox('Did the patient feel itchiness?', map_choice)
    grew = st.sidebar.selectbox('Has the lesion grown over time?', map_choice)
    hurt = st.sidebar.selectbox('Did the patient feel pain?', map_choice)
    changed = st.sidebar.selectbox('Has the lesion changed over time?', map_choice)
    bleed = st.sidebar.selectbox('Has the lesion bleeded before?', map_choice)
    elevation = st.sidebar.selectbox('Is it an elevated lesion?', map_choice)
    
    if image is not None:
        preprocessed_image = image_preprocessor(image)
        input_x1= np.array([preprocessed_image])
        input_x2 = np.array([
            [age, region, itch, grew, hurt, changed, bleed, elevation]
            ])
        input_x2 = scaler.transform(input_x2)

        return input_x1, input_x2

def app():
    st.set_page_config(page_title = 'AI Skin Lesion Classifier', page_icon = icon)
    img = 'https://media.istockphoto.com/photos/melanocytic-nevus-some-of-them-dyplastic-or-atypical-on-a-caucasian-picture-id1188944708?k=20&m=1188944708&s=612x612&w=0&h=eBxqc_iVbVoiVz5673JDjdP8pLNv9S1Zcrk2g0yTaG0='
    st.image(img, width = 450, caption = 'Image from istock.com showing melanocytic nevus.')
    st.header('**Artificial Neural Network Skin Lesion Classifier**')
    st.write('''
    I am an Artifical Neural Network that assists doctors to classify skin lesions into *6 classes*:
    - MEL: Melanoma 
    - SCC: Squamous Cell Carcinoma 
    - NEV: Nevus 
    - ACK: Actinic Keratosis 
    - SEK: Seborrheic Keratosis
    - BCC: Basal Cell Carcinoma
    ''')

    inputs = user_input()
    if inputs is None: 
        st.subheader('Input Data Will Be Displayed Here:')
    else: 
        (input_x1, input_x2) = inputs 
        print(input_x2)
        # Image 
        st.subheader('Uploaded Image:')
        st.write('*Do not worry if the color & resolution changed, that is how I see.*')
        st.image(input_x1)

        # Input Data 
        st.subheader('Input Data:')
        st.write(convert_dataframe(input_x2))

        # Prediction
        st.subheader('Prediction:')
        pred_proba, prediction = make_prediction(model, input_x1, input_x2)
        proba = {
            'Class':['NEV', 'BCC', 'ACK', 'SEK', 'SCC', 'MEL'],
            'Probability': pred_proba[0]
        }
        dataframe = pd.DataFrame(proba)
        fig = px.line_polar(dataframe, r = 'Probability', theta = 'Class', template = 'plotly_dark', line_close = True)
        st.write(dataframe)
        st.plotly_chart(fig)
        st.subheader('*{}* is most likely the diagnosis.'.format(diagnosis[prediction]))

    st.markdown('<a href="{}">The overview of this skin lesion classifier</a>'.format(report_link), unsafe_allow_html = True)

app()
