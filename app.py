import streamlit as st
import requests
import numpy as np
from streamlit_lottie import st_lottie
from PIL import Image
import hydralit_components as hc
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd

st.set_page_config(page_title="Azheimer Predictor", page_icon="")

with hc.HyLoader('Loading..', hc.Loaders.standard_loaders, index=[0]):
    time.sleep(0.5)

def lottie_load(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Inserting lottie animations
lottie = lottie_load("https://assets5.lottiefiles.com/packages/lf20_gkgqj2yq.json")

def local(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
 
def predictor():
    st.title('Azheimer Predictor')
    st.write('## This model predicts if a given scanned image of a brain results Azheimer positive or not.')
    st.write('##### Note: Our model can access real time images too. But inorder to get accurate results, please upload a softcopy.')

    model = tf.keras.models.load_model("customneural.h5")
    image = 0

    option = st.selectbox("How would you like to upload the image?",('Upload a softcopy','Use Camera'))

    if(option=='Upload a softcopy'):
        image = st.file_uploader("Upload a soft copy of your MRI scan.", type=['png', 'jpeg', 'jpg'])
    else:
        image = st.camera_input("Take a picture")
    
    if image != None:
        st.write("## Image you have uploaded")
        img = load_img(image, target_size=(150, 150))
        img = img.resize((150,150))
        st.image(img, width=None)
        img = img_to_array(img)
        img = img.astype('float32')
        img = img / 255.0
        img = np.expand_dims(img,axis=0)
        print(img.shape)
        out = np.squeeze(model.predict(img))

        label1 = ["Positive","Negative"]
        st.write("## Binary Prediction")
        st.write(f"* **Positive** : {out[0]+out[1]+out[3]:.4f}") 
        st.write(f"* **Negative** : {out[2]:.4f}")
        Prob1 = [out[0]+out[1]+out[3], out[2]]
        chart_data1 = pd.DataFrame(
                np.array(Prob1),
                columns=['Probability']
        )
        st.bar_chart(chart_data1)
        
        if(out[0]+out[1]+out[3]>0.4):
            st.write("You might need medical assistance. Contact your nearest physician immediately!")
            label = ['MildDemented', 'ModerateDemented','NonDemented','VeryMildDemented']
            st.write("## Classwise Prediction Analysis(BETA)")
            st.write(f"* **MildDemented** (Class-0): {out[0]:.4f}") 
            st.write(f"* **ModerateDemented** (Class-1): {out[1]:.4f}")
            st.write(f"* **NonDemented** (Class-2): {out[2]:.4f}") 
            st.write(f"* **VeryMildDemented** (Class-3): {out[3]:.4f}")
            
            Prob = [out[0], out[1], out[2], out[3]]
            chart_data = pd.DataFrame(
                    np.array(Prob),
                    columns=['Probability']
            )
            st.bar_chart(chart_data)

predictor()


