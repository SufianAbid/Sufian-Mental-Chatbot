import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

logged_in_user = None


# Initialize Flask app
app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db) # Initialize Flask-Migrate

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Load the chatbot model and data
model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into an array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create a short form for the word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - a matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if the current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by the strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

with app.app_context():
    db.create_all()


@app.route('/get', methods=['GET'])
def get_response():
    if request.method == 'GET':
        message = request.args.get('msg')  # Get the message from the request
        response = chatbot_response(message)  # Call your chatbot function to get a response
        return response

# Route for the home page
@app.route("/")
def home():
    if 'logged_in' in session:
        return render_template("home.html", username=logged_in_user)
    else:
        return redirect(url_for('login'))
# Route for the "About Us" page
@app.route("/about")
def about():
    return render_template("about.html")

# Route for the welcome page
@app.route("/welcome" )
def welcome():
    username = session.get('username')  # Retrieve the username from the session
    return render_template("welcome.html", username=username)

@app.route("/index")
def index_page():
    # You can render the "index.html" template here
    return render_template("index.html")


# Route for registration
@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists in the database
        existing_user = User.query.filter_by(username=username).first()

        if existing_user:
            return "Username already exists. Please choose a different username."

        # If the username doesn't exist, insert the new user into the database
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        # Redirect to the login page after successful registration
        return redirect(url_for('login'))

    return render_template('register.html')

# Route for login

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username and password match a user in the database
        user = User.query.filter_by(username=username, password=password).first()

        if user:
            session['logged_in'] = True
            session['username'] = username  # Store the username in the session
            return redirect(url_for('chatbot'))

    return render_template('login.html')
# Route for the main chatbot page
@app.route('/chatbot')
def chatbot():
    if 'logged_in' in session:
        return render_template('index.html')
    else:
        return redirect(url_for('login'))

@app.route('/')
def index():
    # You can render a template or return a response here
    return redirect(url_for('welcome'))

@app.route('/favicon.ico')
def favicon():
    # Return the favicon file
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

#logout route

@app.route('/logout')
def logout():
    if 'logged_in' in session:
        session.pop('logged_in', None)  # Remove the 'logged_in' session variable
        session.pop('username', None)   # Remove the 'username' session variable
    return redirect(url_for('welcome'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)

