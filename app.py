import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("face_mask_detector.keras")

st.title("Face Mask Detection System")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, use_container_width=True)

    img = image.resize((128, 128))

    img = np.array(img)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("Without Mask")
    else:
        st.success("With Mask")
