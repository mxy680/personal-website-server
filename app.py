from flask import Flask, jsonify, render_template, request
import transformers

app = Flask(__name__)

sentiment_model = transformers.pipeline('sentiment-analysis')


@app.route('/sentiment-analysis', methods=['POST'])
def predict():
    message = request.get_json()['message']
    results = sentiment_model(message)[0]
    prediction = results["label"]
    score = results["score"]
    return jsonify({ 'prediction': prediction, 'score': score })


if __name__ == '__main__':
    # starting app on localhost:8080
    app.run(port=8080)