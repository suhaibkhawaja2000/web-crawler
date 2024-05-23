import requests
import numpy as np
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

app = Flask(__name__)

model = VGG16(weights=None, include_top=True)
model.load_weights('./checkpoints/vgg16_weights.h5')

def classify_image(pil_image):
    pil_image_resized = pil_image.resize((224, 224))
    x = np.array(pil_image_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    result = decode_predictions(preds, top=1)[0][0]
    return result

@app.route('/search', methods=['GET'])
def search_and_classify_images():
    try:
        tag = request.args.get('tag')
        if not tag:
            return jsonify({'error': 'Missing required parameter "tag"'}), 400
        tag = tag.replace(' ', '_')
        url = f"https://www.google.com/search?q={tag}&tbm=isch"
        headers = {'User-Agent': 'My Image Search App'}  
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        images = []
        for img in soup.find_all('img'):
            img_url = img.get('src')
            if img_url and img_url.startswith('http'):
                images.append(img_url)
                if len(images) == 5:
                    break
        
        results = []
        for img_url in images:
            img_response = requests.get(img_url)
            img = Image.open(BytesIO(img_response.content))
            label = classify_image(img)
            results.append({
                'url': img_url,
                'user tag':tag,
                'model tag' : label[1],
            })
        return jsonify({'results': results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods = ['POST'])
def upload_and_classify_images():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    image = request.files['image']
    img = Image.open(image)
    label = classify_image(img)
    return jsonify({'classification': label[1]})


if __name__ == '__main__':
    app.run(debug=True)
