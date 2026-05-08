import streamlit as st
from PIL import Image
import numpy as np

# Import TFLite runtime
from tflite_runtime.interpreter import Interpreter

# Load model
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Title
st.title("Face Mask Detection System")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, use_container_width=True)

    img = image.resize((128, 128))

    img = np.array(img, dtype=np.float32)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    if prediction[0][0] > 0.5:
        st.error("Without Mask")
    else:
        st.success("With Mask")
