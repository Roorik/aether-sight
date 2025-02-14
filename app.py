import streamlit as st

from camera_input_live import camera_input_live
from backend.service.recognizer import Recognizer

"# Streamlit camera input live Demo"
"## Try holding a qr code in front of your webcam"

image = camera_input_live(debounce=2000)
if image is not None:
    Recognizer().execute(image)
    st.image(image)

