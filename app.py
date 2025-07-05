import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model.keras")

def predict(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    result = "Dog" if prediction > 0.5 else "Cat"
    return result


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Dog vs Cat Classifier",
    description="Upload an image to predict whether it's a Dog or a Cat"
)


interface.launch()
