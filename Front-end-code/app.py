import streamlit as st
import torch
from PIL import Image
import numpy as np
import plotly.express as px

st.title("Brain Tumor Detector")
st.subheader("Paste the MRI scan image below to detect if there is a tumor or not")
st.text("Github repository: https://github.com/DEboy2007/Brain-Tumor-Detector")
image = st.file_uploader("Upload MRI scan", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False)

if image:
    image = Image.open(image)
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", device="cpu")
    results = model(image, size=640)
    fig = px.imshow(np.squeeze(results.render()), aspect="equal")

    st.plotly_chart(fig)
