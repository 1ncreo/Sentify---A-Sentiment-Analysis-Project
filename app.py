# Library imports
from flask import Flask, render_template, request
from googleapiclient.discovery import build
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords


# Initialize Flask app
app = Flask(__name__)

# YouTube API key
api_key = 'ADD_YOUR_KEY'

# Function to extract video ID from YouTube URL


def get_video_id(url):
    return url.split("v=")[1]

# Function to fetch all comments from a YouTube video


def get_all_comments(video_url):
    video_id = get_video_id(video_url)
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    nextPageToken = None
    while True:
        comments_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            pageToken=nextPageToken
        ).execute()
        comments.extend(comments_response['items'])
        nextPageToken = comments_response.get('nextPageToken')
        if not nextPageToken:
            break
    print(f"Number of comments: {len(comments)}")
    comment_data = []
    for comment in comments:
        comment_data.append({
            'comment_text': comment['snippet']['topLevelComment']['snippet']['textDisplay'],
        })
    comments_df = pd.DataFrame(comment_data)
    return comments_df

# Function to preprocess text


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text

# Function to classify sentiments


def classify_sentiments(sentiments):
    classified_sentiments = []
    for sentiment in sentiments:
        if sentiment < 0.4:
            classified_sentiments.append("Negative")
        elif sentiment >= 0.4 and sentiment < 0.7:
            classified_sentiments.append("Neutral")
        else:
            classified_sentiments.append("Positive")
    return classified_sentiments

# Route for home page


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/anal.html', methods=['GET', 'POST'])
def anal():
    percentages = None
    if request.method == 'POST':
        video_url = request.form['video_url']
        comments_df = get_all_comments(video_url)
        unseen_reviews = comments_df['comment_text']
        unseen_processed = []
        for review in unseen_reviews:
            review = preprocess_text(review)
            unseen_processed.append(review)

        # Tokenize the text using Keras Tokenizer
        tokenizer = Tokenizer()  # Create tokenizer instance
        tokenizer.fit_on_texts(unseen_processed)  # Fit tokenizer on data
        unseen_tokenized = tokenizer.texts_to_sequences(
            unseen_processed)  # Tokenize text

        # Padding sequences
        maxlen = 100
        unseen_padded = pad_sequences(
            unseen_tokenized, padding='post', maxlen=maxlen)

        # Load pretrained LSTM model
        pretrained_lstm_model = load_model('/root/Sentify/c1_lstm_model_acc_0.863.h5')
        unseen_sentiments = pretrained_lstm_model.predict(unseen_padded)
        classified_sentiments = classify_sentiments(
            [sent[0] for sent in unseen_sentiments])

        # Calculate percentage of each sentiment category
        num_comments = len(classified_sentiments)
        num_positive = classified_sentiments.count("Positive")
        num_neutral = classified_sentiments.count("Neutral")
        num_negative = classified_sentiments.count("Negative")

        percentage_positive = (num_positive / num_comments) * 100
        percentage_neutral = (num_neutral / num_comments) * 100
        percentage_negative = (num_negative / num_comments) * 100

        percentages = {
            "Positive": percentage_positive,
            "Neutral": percentage_neutral,
            "Negative": percentage_negative
        }

    return render_template('anal.html', percentages=percentages)


if __name__ == '__main__':
    app.run(debug=True)
