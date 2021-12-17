import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
st.title("Sign Language Classification")
st.header("Gesture Model")
image_path = st.file_uploader("Image")
if image_path is not None:
    image = Image.open(image_path)
    st.image(image)


@st.cache(hash_funcs={'tensorflow.python.keras.utils.object_identity.ObjectIdentityDictionary': id}, allow_output_mutation=True)
def model_upload():
    model = load_model("C:/Users/harsh/PycharmProjects/Harshvir_S/gestures.h5")
    print("Loading")
    return model


def predict(model):
    img = np.array(image)
    img = cv2.resize(img, (100, 100))
    img = img / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    pred = np.argmax(prediction, axis=-1)
    return pred


if image_path is not None:
    m = model_upload()
    p = predict(m)
    st.text(p)
