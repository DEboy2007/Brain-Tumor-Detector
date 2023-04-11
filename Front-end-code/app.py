import streamlit as st
import torch
from PIL import Image
import numpy as np
import plotly.express as px

st.title("Brain Tumor Detector")
st.write("By Dhanush Ekollu")
st.write("""##### This project detects suspected tumors in MRI brain scans.\n
**Github repository**: https://github.com/DEboy2007/Brain-Tumor-Detector""")
st.write(
    "#### Paste the MRI scan image below to detect possible tumors")
image = st.file_uploader("Upload MRI scan", type=[
                         "jpg", "jpeg", "png", "webp"], accept_multiple_files=False)

if image:
    image = Image.open(image)
    model = torch.hub.load("ultralytics/yolov5", "custom",
                           path="Front-end-code/best.pt", device="cpu")
    results = model(image, size=640)
    fig = px.imshow(np.squeeze(results.render()), aspect="equal")

    st.plotly_chart(fig)

st.subheader("Credits:")
st.write("""Dataset used: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection
Annotations done by me""")
