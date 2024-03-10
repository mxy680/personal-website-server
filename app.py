from flask import Flask, jsonify, render_template, request
import transformers

app = Flask(__name__)

model = transformers.pipeline('sentiment-analysis')
    
def get_prediction(message,model):
    # inference
    results = model(message)  
    return results


@app.route('/', methods=['POST'])
def predict():
    message = request.get_json()['message']
    results = get_prediction(message, model)
    prediction = results[0]["label"]
    score = results[0]["score"]
    return jsonify({ 'prediction': prediction, 'score': score })


if __name__ == '__main__':
    # starting app on localhost:8080
    app.run(port=8080)