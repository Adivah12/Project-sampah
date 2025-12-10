import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS # Wajib untuk akses dari website luar
from PIL import Image # Pengganti tensorflow.keras.preprocessing
import io

# --- IMPORT TFLITE ---
# Kita gunakan try-except agar fleksibel.
# Di Lokal (PC) pakai tensorflow biasa gapapa.
# Di Render (Cloud) kita paksa pakai tflite_runtime biar ringan.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)
CORS(app) # Izinkan website lain mengakses API ini

# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = 'model_final.tflite'
CLASS_NAMES = ['Anorganik', 'Organik']

print("Loading model...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
# Biasanya format (1, 224, 224, 3) -> ambil tinggi dan lebar
height = input_shape[1]
width = input_shape[2]
print(f"Model loaded. Input shape: {height}x{width}")

def prepare_image(image_bytes):
    """Fungsi helper untuk memproses gambar dari memori"""
    # 1. Buka gambar dari bytes (tanpa simpan ke disk)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 2. Resize sesuai model
    img = img.resize((width, height))
    
    # 3. Ubah ke Array
    img_array = np.array(img)
    
    # 4. Tambah dimensi batch (1, H, W, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Normalisasi Float32
    # PENTING: Cek ulang cara training Anda. 
    # Jika pakai rescale 1./255, baris di bawah WAJIB NYALA.
    if input_details[0]['dtype'] == np.float32:
        img_array = img_array.astype(np.float32) / 255.0 
        
    return img_array

def run_inference(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Logika Binary vs Categorical
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

# --- ROUTE 1: Untuk API (Website Luar / Mobile App) ---
@app.route('/predict-api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    
    try:
        # Proses langsung di memori
        img_array = prepare_image(file.read())
        label, confidence = run_inference(img_array)
        
        return jsonify({
            "status": "success",
            "prediction": label,
            "confidence": f"{confidence*100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ROUTE 2: Untuk Halaman Web Sederhana (Demo) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file"
        file = request.files['file']
        if file.filename == '':
            return "No filename"
            
        # Untuk tampilan web demo, kita butuh file dibaca dua kali:
        # 1. Untuk prediksi (read bytes)
        # 2. Untuk ditampilkan (disimpan sementara ke folder static)
        
        # Baca bytes untuk prediksi
        file_bytes = file.read() 
        img_array = prepare_image(file_bytes)
        label, conf = run_inference(img_array)
        
        # Simpan ke disk HANYA untuk ditampilkan di HTML (Opsional)
        # Reset pointer file agar bisa disave
        file.seek(0) 
        if not os.path.exists('static/uploads'):
            os.makedirs('static/uploads')
        save_path = os.path.join('static/uploads', file.filename)
        file.save(save_path)
        
        return render_template('index.html', 
                             prediction=label, 
                             confidence=f"{conf*100:.2f}%", 
                             image_path=save_path)
                             
    return render_template('index.html')

if __name__ == '__main__':
    # Debug=True matikan saat production!
    app.run(debug=True, host='0.0.0.0', port=5000)