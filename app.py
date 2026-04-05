import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model using the modern Keras 3 format
print("Loading model...")
model = tf.keras.models.load_model('alzheimer_model.keras', compile=False)
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
print("Model loaded successfully!")

def generate_gradcam_and_predict(img_path, filename):
    # 1. Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # Preprocess specifically for ResNet50
    preprocessed_img = preprocess_input(img_array_expanded.copy())

    # 2. Get Prediction
    preds = model.predict(preprocessed_img)
    pred_index = np.argmax(preds[0])
    predicted_class = class_names[pred_index]
    confidence = float(preds[0][pred_index]) * 100

    # 3. Grad-CAM Setup (Using your architecture)
    base_model = model.get_layer('resnet50')
    last_conv_layer = base_model.get_layer('conv5_block3_out')
    
    grad_model = tf.keras.Model([base_model.inputs], [last_conv_layer.output, base_model.output])
    
    # Build Top Model (everything after resnet50)
    top_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = top_input
    resnet_idx = model.layers.index(base_model)
    for layer in model.layers[resnet_idx+1:]:
        x = layer(x)
    top_model = tf.keras.Model(top_input, x)

    # 4. Compute Gradients
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(preprocessed_img)
        tape.watch(conv_outputs)
        preds_top = top_model(base_outputs)
        class_channel = preds_top[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # 5. Mask and Superimpose Heatmap (Clean Background)
    img_cv = cv2.imread(img_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    
    # Create a brain mask to ignore black background
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, brain_mask = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
    
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_masked = heatmap_resized * brain_mask
    
    heatmap_color = np.uint8(255 * heatmap_masked)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)
    
    brain_mask_3d = np.repeat(brain_mask[:, :, np.newaxis], 3, axis=2)
    heatmap_color = heatmap_color * brain_mask_3d
    
    superimposed_img = heatmap_color * 0.4 + img_cv
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Save the heatmap image to the static folder so the website can show it
    heatmap_filename = "heatmap_" + filename
    heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
    cv2.imwrite(heatmap_path, superimposed_img)

    return predicted_class, confidence, heatmap_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run model & Grad-CAM
        predicted_class, confidence, heatmap_path = generate_gradcam_and_predict(filepath, filename)

        return jsonify({
            'class': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'original_image': filepath,
            'heatmap_image': heatmap_path
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)