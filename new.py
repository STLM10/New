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
    image = load_img(uploaded_file, grayscale=True, target_size=(50,50,1))
    st.write(image)
    #img = rgb_to_grayscale(image)
    img = img_to_array(image)
#     st.write(img)
    img1 = np.asarray(img)
#     st.write(img1)
    img1 = img1 / 255.0
    st.write(img1.shape)
    img1 = img1.flatten()
    img2 = img1.reshape(1,-1)
#     img1 = np.array(img1)
#     img1 = img1.reshape((1,2500))
    a=model.predict(img2)
    st.write(a)
    if a<0.5:
        st.text('cat')
    else:
        st.text('dog')
