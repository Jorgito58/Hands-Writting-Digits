import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np
from annotated_text import annotated_text 
from PIL import Image
from streamlit_option_menu import option_menu
import time

#Loading Model
model_new = keras.models.load_model('D:\\Software Development\\PythonLearning\\Curso-Python\\0. Python_Proyects\\HandsWritting_Numbers\\saved_model\\handsw_conv_model')


image = Image.open("ia.jpg")
st.set_page_config(page_title='Dashboard', page_icon = image, layout='wide')

with st.sidebar:
    selected = option_menu(
        menu_title = "",
        options = ["Software", "Code"],
        icons = ["house", "book"],
        menu_icon = "cast",
        default_index = 0,
        orientation = "vertical",
    )
    st.write("By: 💻Wise_George")
    with st.spinner("Loading..."):
            time.sleep(2)
            succ = st.success("✅Done!")
            time.sleep(1)
            succ.empty()

    

if selected == "Software":
    col1, col2, col3 =st.columns([3,9,3])
    with col2:
        st.title("Hand Writting Digit Recognizer")
        annotated_text(("Try to Write Carefully","","#000"))
        st.header("")

        SIZE = 192

        canvas_result = st_canvas(
            fill_color="#ffffff",
            stroke_width=10,
            stroke_color='#ffffff',
            background_color="#000000",
            height=150,width=150,
            drawing_mode='freedraw',
            key="canvas",
        )

        if canvas_result.image_data is not None:
            img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
            img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            st.write('Input Image')
            st.image(img_rescaling)

        if st.button('Predict'):
            test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pred = model_new.predict(test_x.reshape(1, 28, 28, 1))
            numbs = ['0️⃣','1️⃣','2️⃣','3️⃣','4️⃣','5️⃣','6️⃣','7️⃣','8️⃣','9️⃣']
            var = np.argmax(pred[0])
            st.header("Result: {}".format(numbs[var]))
            st.header("")
            st.bar_chart(pred[0])

if selected =="Code":
    st.code("# Working On")
