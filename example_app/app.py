from model_and_interface.model_interface import predict_species

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'example_app/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def analyze_image(image_path):
    # Example: return image format and size
    return predict_species(image_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            result = analyze_image(filepath)
            return render_template('index.html', result=result)
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)