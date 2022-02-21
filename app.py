import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

st.write('''
# A Pineapple Ripeness Index Classifier
''')
st.write("This is a web application that can detect a Pineapple Ripeness")

file = st.file_uploader("Please upload an Image of a Pineapple", type=['jpg','png'])


def predict_stage(image_data,model):
    size = (224, 224)
    image = ImageOps.fit(image_data,size, Image.ANTIALIAS)
    image_array = np.array(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    preds = ""
    prediction = model.predict(data)
    if np.argmax(prediction)==0:
        st.write(f"Fully Ripe")
        st.write(f"Not advisable to export and sell. Short shelf life")
    elif np.argmax(prediction)==1:
        st.write(f"Partially Ripe")
        st.write(f"Perfectly viable to export and sell")
    else :
        st.write(f"Unripe")
        st.write(f"Not yet ripe, not recommended for harvest")

    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    model = tf.keras.models.load_model('pineappleindex.h5')
    Generate_pred = st.button("Predict Ripeness Stage..")
    if Generate_pred:
        prediction = predict_stage(image, model)
        st.text("Probability (0: Fully Ripe, 1: Partially Ripe, 2: Unripe")
        st.write(prediction)