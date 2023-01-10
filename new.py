import streamlit as st
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.image import rgb_to_grayscale
import numpy as np
import pickle
model = pickle.load(open('LR.pkl','rb'))
st.title('Image Classifier')


uploaded_file = st.file_uploader('Drop the image')



if st.button('check'):
    image = load_img(uploaded_file, grayscale=True, target_size=(50,50))
    #img = rgb_to_grayscale(image)
    img = img_to_array(image)
    img1 = np.array(img)
    img1 = img1 / 255.0
    img1 = img1.resize((50,50))
    img1 = img1.reshape(1,2500)
    a=model.predict(img1)
    if a<0.5:
        st.text('cat')
    else:
        st.text('dog')
