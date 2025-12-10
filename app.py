import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

# --- IMPORT TFLITE ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)
CORS(app)

# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = 'model_final.tflite'
CLASS_NAMES = ['Anorganik', 'Organik']

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]


def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((width, height))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if input_details[0]['dtype'] == np.float32:
        img_array = img_array.astype(np.float32) / 255.0

    return img_array


def run_inference(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if output_data.shape[-1] == 1:
        score = float(output_data[0][0])
        if score > 0.5:
            label = CLASS_NAMES[1]
            confidence = score
        else:
            label = CLASS_NAMES[0]
            confidence = 1 - score
    else:
        score = output_data[0]
        label_index = np.argmax(score)
        label = CLASS_NAMES[label_index]
        confidence = float(np.max(score))

    return label, confidence


# ======================================================
#     API ONLY — NO HTML
# ======================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "ML API is running"}), 200


@app.route('/predict-api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']

    try:
        img_array = prepare_image(file.read())
        label, confidence = run_inference(img_array)

        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
