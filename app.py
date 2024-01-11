from flask import render_template, Flask, request
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

app = Flask(__name__)

SKIN_CLASSES = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'Basal Cell Carcinoma',
  2: 'Benign Keratosis',
  3: 'Dermatofibroma',
  4: 'Melanoma',
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion'
}

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path = 'static/data/' + f.filename
        f.save(path)

        jason_file = open('modelnew.json', 'r')
        loaded_json_model = jason_file.read()
        jason_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights('modelnew.h5')

        ip_img = image.load_img(path, target_size=(224, 224))
        ip_img = image.img_to_array(ip_img)
        ip_img = np.expand_dims(ip_img, axis=0)
        ip_img = preprocess_input(ip_img)

        prediction = model.predict(ip_img)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]

        tf.keras.backend.clear_session()

        return render_template('uploaded.html', title='Success', predictions=disease, acc=accuracy * 100, img_file=f.filename)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
