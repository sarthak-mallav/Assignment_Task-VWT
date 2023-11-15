from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class AnalyzeSentiment(Resource):
    def __init__(self):
        super(AnalyzeSentiment, self).__init__()
        self.tokenizer = Tokenizer(num_words=1000)
        self.actual_max_sequence_length = 61  
        self.model = load_model('sen.h5')

    def post(self):
        data = request.get_json()
        text = data.get('text')

        if not (2 <= len(text) <= 150):
            return jsonify({'error': 'Text length should be between 2 and 150 characters.'}), 400

        result = self.predict_sentiment(text)
        return jsonify(result)

    def predict_sentiment(self, text):
        sequences = self.tokenizer.texts_to_sequences([text])
        padded_sequences = pad_sequences(sequences, maxlen=self.actual_max_sequence_length, dtype='int32', value=0)
        sentiment = self.model.predict(padded_sequences, batch_size=1, verbose=1)[0]

        predicted_classification = self.get_predicted_classification(sentiment[0])
        print(sentiment)

        result = {
            "Text": text,
            "Sentiment": predicted_classification
        }

        return result

    def get_predicted_classification(self, sentiment):
    	if (sentiment <= 0.0).any():
        	return 'Sadness'
    	elif (sentiment <= 0.1).any():
        	return 'Anger'
    	elif (sentiment <= 0.2).any():
        	return 'Love'
    	elif (sentiment <= 0.3).any():
        	return 'Surprise'
    	elif (sentiment <= 0.4).any():
        	return 'Fear'
    	else:
        	return 'Joy'

api.add_resource(AnalyzeSentiment, '/predict_sentiment')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

