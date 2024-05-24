from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model("models/resnet_model_1.h5", compile=False)

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.route("/predict", methods=['POST'])
def predict():
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(request.files['file'])
    image = np.array(image.resize((256, 256)))[:, :, :3]

    img_batch = np.expand_dims(image, 0)

    predictions = model.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == '__main__':
    # app.debug = True
    app.run(host='localhost', port=8000)
