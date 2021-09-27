import tensorflow_hub as hub
import streamlit as st
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tf_utils import convert_to_img_tensor, tensor_to_image
import os

st.markdown('# Artistic Style Transfer')

#########################################
#
# FAST STYLE TRANSFER
#
#########################################

st.markdown('## Fast Style Transfer')
st.markdown('The goal of artistic style transfer is to impose one image\'s \
   style onto another such as imposing starry night onto a selfie. \
   You can try it below! Upload a source image, and a style image and watch \
   the magic happen!')

ex_left, ex_middle, ex_right = st.columns(3)
ex_left.image('images/prof_pic.jpg', caption='source image')
ex_middle.image('images/starry_night.jpeg', caption='target style')
ex_right.image('images/stylized-image.png', caption='stylized prediction')

st.markdown('FYI, images are processed with a single max dimension of 512, \
   so ideally you want both your images to be 512 by 512 for the best \
   image quality or as close as a square as possible since aspect ratio \
   is maintained.')

MODEL_URL = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

for key in ['source_image', 'style_image', 'stylized_image']:
    if key not in st.session_state:
        st.session_state.source_image = None
        st.session_state.style_image = None
        st.session_state.stylized_image = None


form = st.form('fast-style-transfer-form')
st.session_state.source_image_data = form.file_uploader(
    'Choose a source image to recieve a style',
    type=['png', 'jpg', 'jpeg'])


st.session_state.style_image_data = form.file_uploader(
    'Choose an image to to get a style from', type=['png', 'jpg', 'jpeg'])


def predict_fast_stylized_image():
    with st.spinner('Making prediction...'):
        if st.session_state.source_image_data is None:
            form.error('Please provide a valid Source Image')
            return
        if st.session_state.style_image_data is None:
            form.error('Please provide a valid Style Image')
            return

        with Image.open(st.session_state.source_image_data) as im:
            source_image = im.convert('RGB')

        with Image.open(st.session_state.style_image_data) as im:
            style_image = im.convert('RGB')

        hub_model = hub.load(MODEL_URL)
        stylized_image = hub_model(
            tf.constant(convert_to_img_tensor(source_image)),
            tf.constant(convert_to_img_tensor(style_image)))[0]
        st.session_state.stylized_image = tensor_to_image(stylized_image)


clicked = form.form_submit_button('Predict Stylized Image')

if clicked:
    if st.session_state.source_image_data is None:
        form.error('Please provide a valid Source Image')
        
    elif st.session_state.style_image_data is None:
        form.error('Please provide a valid Style Image')
    else:  
        predict_fast_stylized_image()

if st.session_state.stylized_image is not None:
    _, result, _ = st.columns((1, 3, 1))
    img = st.session_state.stylized_image
    img_height, img_width = img.size
    result.image(img, caption=f"{img_height} x {img_width}")

    image_buffer = BytesIO()
    img.save(image_buffer, format='png')

    _ , download, _ = st.columns((1, 3, 1))
    download.download_button(
        'Download Stylized Image', image_buffer,
        file_name='stylized-image.png', mime='image/png')


#----------------------Hide Streamlit footer----------------------------
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#--------------------------------------------------------------------
