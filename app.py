import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model.keras")

def predict(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]
    class_names = ["Cat", "Dog", "Others"]
    pred_class = np.argmax(prediction)
    label = class_names[pred_class]

    return f"<div style='text-align:center; font-size:24px; font-weight:bold'>{label}</div>"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.HTML(),
    title="Cat vs Dog vs Others Classifier",
    description="Upload an image to predict whether it's a Cat, Dog, or something else (Others)"
)

interface.launch()
