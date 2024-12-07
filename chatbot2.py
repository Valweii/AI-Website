import openai
from flask import Flask, request, jsonify, render_template, url_for
import nltk
import numpy
import tflearn
import tensorflow
import random
import json
import os
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv

# Ensure you download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Ensure the script's directory is the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the dataset
with open("./data/intents.json") as file:
	data = json.load(file)

# Initialize Flask app
app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Set OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
	for pattern in intent["patterns"]:
		wrds = nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_x.append(wrds)
		docs_y.append(intent["tag"])

	if intent["tag"] not in labels:
		labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)


training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
	bag = []

	wrds = [stemmer.stem(w) for w in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else:
			bag.append(0)
	output_row = out_empty[:]
	output_row[labels.index(docs_y[x])] = 1

	training.append(bag)
	output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

base = tflearn.DNN(net)

try:
	base.load("base.tflearn")
except:
	base = tflearn.DNN(net)
	base.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
	base.save("base.tflearn")
 
def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]


	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

test_patterns = []
test_labels = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        test_patterns.append(pattern)  # Add the pattern
        test_labels.append(intent["tag"])  # Add the corresponding label

# Route for the chatbot
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    inp = request.get_json()
    if not inp or 'message' not in inp:
        return jsonify({"response": "Invalid input. Please provide a message."}), 400

    user_input = inp['message']

    # Predict intent
    results = base.predict([bag_of_words(user_input, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    test_bags = [bag_of_words(pattern, words) for pattern in test_patterns]	
    predictions = [labels[numpy.argmax(base.predict([bag])[0])] for bag in test_bags]
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Check confidence threshold
    if results[results_index] > 0.9:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                return jsonify({"response": random.choice(responses)})
    else:
        try:
            openai_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=100
            )
            gpt_response = openai_response['choices'][0]['message']['content']
            return jsonify({"response": gpt_response})
        except Exception as e:
            return jsonify({"response": f"Error communicating with OpenAI API: {str(e)}"}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000)
