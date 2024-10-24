from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)


model = load_model('my_model.h5')

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image'] 

        
        img = Image.open(file) 
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0  

        
        if img_array.shape[-1] == 3:
            img_array = img_array.reshape(1, 224, 224, 3)  
        else:
            return render_template('index.html', prediction="Invalid image format. Please upload an RGB image.")

        
        prediction_prob = model.predict(img_array)  
        prediction = "Yes" if prediction_prob[0][0] > 0.5 else "No" 

        return render_template('index.html', prediction=prediction)  

if __name__ == '__main__':
    app.run(debug=True)  
