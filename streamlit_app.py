import streamlit as st
import requests
import io
from PIL import Image

st.set_page_config(page_title="BLIP Image Captioning", layout="wide")
st.title("BLIP Image Captioning")
st.write("Upload any image to generate a caption using BLIP")

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with st.spinner("Generating caption..."):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        response = requests.post(API_URL, data=buf.getvalue())
        if response.status_code == 200:
            result = response.json()
            caption = result[0].get("generated_text", "No caption") if isinstance(result, list) else str(result)
        else:
            caption = "Error: " + str(response.status_code)
    with col2:
        st.success("Caption Generated!")
        st.markdown("### " + caption)