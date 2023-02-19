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

st.set_page_config(page_title="Main Page", page_icon=":tada:", layout="wide")

#inculding comments
with hc.HyLoader('Loading..', hc.Loaders.standard_loaders, index=[0]):
    time.sleep(0.5)

# specify the primary menu definition
# menu_data = [
#     {'icon': "far fa-copy", 'label': "Conditions"},
#     {'id': 'Copy', 'label': "Well-Being"},
#     {'icon': "far fa-chart-bar", 'label': "Symptom Checker"},  # no tooltip message
#     {'icon': "far fa-address-book", 'label': "Find doctors nearby"},
#     {'id': 'options','label': "More"},
#     {'icon': "fas fa-tachometer-alt", 'label': "Dashboard", 'ttip': "See the history here"},
#     # can add a tooltip message

# ]
# # we can override any part of the primary colors of the menu
# # over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
# over_theme = {'txc_inactive': '#FFFFFF', 'menu_background':'black','txc_active':'white','option_active':'blue'}
# menu_id = hc.nav_bar(menu_definition=menu_data, home_name='Main Page', override_theme=over_theme)

# # get the id of the menu item clicked
# st.info(f"{menu_id=}")


def lottie_load(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Inserting lottie animations

lottie = lottie_load("https://assets5.lottiefiles.com/packages/lf20_gkgqj2yq.json")

# img_link1 = Image.open(r"C:\Users\Harinee\Desktop\Training_120178.jpg")
# img_link2 = Image.open(r"C:\Users\Harinee\Desktop\exp.png")
# img=Image.open(r"C:\Users\Harinee\Desktop\alz12328-fig-0015-m.webp")

# Adding local CSS files

def local(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


#local(r'C:\Users\kadal\Documents\HAXIOS\Style\content.css')

# organises the code


add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

def main_page():

    with st.container():
        st.subheader("An Overview")
        st.title("Alzheimer’s disease is a type of brain disease caused by damage to nerve cells (neurons) in the brain.")
        st.write("The neurons damaged first are those in parts of the brain responsible for memory, language and thinking. As a result, the first symptoms of Alzheimer’s disease tend to be memory, language and thinking problems.")
        st.write("check if you are affected by alzheimer's with our website")
        with st.container():
            st.write("---")  # divider
            left_column, right_column = st.columns(2)  # inserting 2 columns
            with left_column:
                st.header("What I do")
                st.write("##")
                st.write(
                    """
                Alzheimer's disease is the most common type of dementia. It is a progressive disease beginning with mild
                memory loss and possibly leading to loss of the ability to carry on a conversation and respond to the environment.
                Alzheimer's disease involves parts of the brain that control thought, memory, and language.""")

                st.write(
                    "[Click here to learn more>](https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447)")

            with right_column:
                # insert the animation
                st_lottie(lottie, height=300, key="Perfect Cure")

        with st.container():
            st.write("---")
            left_column, right_column = st.columns(2)

            with left_column:
                st.header("How do we plan to implement this")
                st.write('##')
                st.write("Do you want to check if you have alzymers disease??")
                st.write("This can be done by a single click of the mouse and 1 image upload")
            with right_column:
                st.write("Click here to checkout whether you have alzymers disease")
                res = st.button("Click here")
                if res:
                    st.write(":smile:")
                    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "csv", "png"])

                    if uploaded_file is not None:
                        # Process the uploaded file
                        # contents = uploaded_file.read()
                        img = Image.open(uploaded_file)
                        st.write("You uploaded the following file:")
                        st.image(img, "Uploaded_Image")

        with st.container():
            st.write("--")
            st.header("How deadly is the disease")
            st.write("##")
            img_col, text_col = st.columns((1, 2))  # tex t column is twice as big as image

            # with img_col:  # inserting images
            #     st.image(img_link1)
            #     st.image(img_link2)

            with text_col:
                st.write(
                    """
        It is the fifth leading cause of death for adults aged 65 and older, and the seventh leading cause of death for all adults.
        Alzheimer's disease involves partsof the brain that control thought, memory, and language, and, over time,
        can seriously affect a person's ability to carry out daily activities.""")

            # link to videos
            st.markdown("Watch video...](https://www.youtube.com/watch?v=7F-t9yvPP_0)")

        # __CONTACT BLOCK__

        with st.container():
            st.write("--")  # divider
            st.header("Contact Us")
            st.write("##")  # extra space

            contact_form = """
            <form action="https://formsubmit.co/srinidhi.k2021@vitstudent.ac.in" method="POST">

            <input type="hidden" name="_capta" value="false">
             <input type="text" name="name" placeholder="Your name"required>
             <input type="email" name="email" placeholder="Your email" required>
             <textarea name="message" placeholder="Your Message here" required></textarea>

             <button type="submit">Send</button>
            </form>
            """
            left_col, right_col = st.columns(2)

            # eject html code into web app
            with left_col:
                st.markdown(contact_form, unsafe_allow_html=True)
            with right_col:
                st.empty()

    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")


def page2():
    st.markdown("# Page 2 ❄️")
    with st.container():
        st.title("A Brief Look")
        st.subheader("Do you know the statistics of how many people are affected worldwide? Let's have a look!")
        st.write("--")
        img_col, text_col = st.columns((2,1))  # tex t column is twice as big as image
        # with img_col:  # inserting images
        #      st.image(img)

        with text_col:
            st.write("This is the statistical representation of alzheimer's disease reported world-wide")
            st.write("As there could be nu")



    st.sidebar.markdown("# Page 2 ❄️")


def page3():
    st.write("Upload")
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")
    st.title('Azheimer Predictor')
    st.write('## Azheimer Classification')

    st.write("Due to the unavailability of huge data, I used transfer learning with VGG16.")
    st.write("For this problem even simpler pretrained networks would give pretty decent results.")

    model = tf.keras.models.load_model("customneural.h5")
    image = st.file_uploader("Upload Image to classify as Rural/Urban Scene", 
    type=['png', 'jpeg', 'jpg'])


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


        label = ['MildDemented', 'ModerateDemented','NonDemented','VeryMildDemented']
        st.write("## Classwise Prediction")
        st.write(f"* **MildDemented** (Class-0): {out[0]:.4f}") 
        st.write(f"* **ModerateDemented** (Class-1): {out[1]:.4f}")
        st.write(f"* **NonDemented** (Class-2): {out[2]:.4f}") 
        st.write(f"* **VeryMildDemented** (Class-3): {out[3]:.4f}")
        print(out)
        Prob = [out[0], out[1], out[2], out[3]]
        chart_data = pd.DataFrame(
                np.array(Prob),
                columns=['Probability']
        )
        st.bar_chart(chart_data)



page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
    "Page 3": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


