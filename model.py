import pickle
import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def nltk_setup():
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    
class Constants:
    ROOT_PATH = "models/"
    MODEL_NAME = "sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    # DATA_DIR = "./data/"
    INPUT_FILE = "data/sample30.csv"


class SentimentRecommender:
    def __init__(self):
        self.data = self.load_data(Constants.INPUT_FILE)

        self.model = self.load_model(Constants.MODEL_NAME)
        self.vectorizer = self.load_model(Constants.VECTORIZER)
        self.user_final_rating = self.load_model(Constants.RECOMMENDER)
        self.cleaned_data = self.load_model(Constants.CLEANED_DATA)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def load_model(self, param):
        """loads pickle objects"""
        object_path = Constants.ROOT_PATH + param
        return pickle.load(open(object_path, "rb"))

    def load_data(self, param):
        """loads input data"""
        return pd.read_csv(param)

    def getRecommendationByUser(self, user):
        """function to get the top product 20 recommendations for the user"""
        return list(
            self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        )

    def getSentimentRecommendations(self, user):
        """function to filter the product recommendations using the sentiment model and get the top 5 recommendations"""
        # validate if user exists
        if user in self.user_final_rating.index:
            recommendations = list(
                self.user_final_rating.loc[user]
                .sort_values(ascending=False)[0:20]
                .index
            )
            filtered_data = self.cleaned_data[
                self.cleaned_data.id.isin(recommendations)
            ]

            # preprocess the text and predict sentiment
            X = self.vectorizer.transform(
                filtered_data["lemmatized_reviews"].values.astype(str)
            )
            filtered_data["predicted_sentiment"] = self.model.predict(X)
            temp = filtered_data[["id", "predicted_sentiment"]]
            temp_grouped = temp.groupby("id", as_index=False).count()
            temp_grouped["pos_review_count"] = temp_grouped.id.apply(
                lambda x: temp[(temp.id == x) & (temp.predicted_sentiment == 1)][
                    "predicted_sentiment"
                ].count()
            )
            temp_grouped["total_review_count"] = temp_grouped["predicted_sentiment"]
            temp_grouped["pos_sentiment_percent"] = np.round(
                temp_grouped["pos_review_count"]
                / temp_grouped["total_review_count"]
                * 100,
                2,
            )
            sorted_products = temp_grouped.sort_values(
                "pos_sentiment_percent", ascending=False
            )[0:5]
            return (
                pd.merge(self.data, sorted_products, on="id")[
                    ["name", "brand", "manufacturer", "pos_sentiment_percent"]
                ]
                .drop_duplicates()
                .sort_values(["pos_sentiment_percent", "name"], ascending=[False, True])
            )

        else:
            print(f"User name {user} doesn't exist")
            return None

    def classify_sentiment(self, review_text):
        """function to classify the sentiment to 1/0 - positive or negative - using the trained ML model"""
        review_text = self.clean_text(review_text)
        X = self.vectorizer.transform([review_text])
        y_pred = self.model.predict(X)
        return y_pred

    def clean_text(self, text):
        """function to preprocess the text before it's sent to ML model"""
        text = text.lower().strip()
        text = re.sub("\[\s*\w*\s*\]", "", text)
        dictionary = "abc".maketrans("", "", string.punctuation)
        text = text.translate(dictionary)
        text = re.sub("\S*\d\S*", "", text)

        # remove stop-words and convert it to lemma
        text = self.lemmatize_text(text)
        return text

    def get_wordnet_pos(self, tag):
        """function to get the pos tag to derive the lemma form"""
        if tag.startswith("J"):
            return wordnet.ADJ
        elif tag.startswith("V"):
            return wordnet.VERB
        elif tag.startswith("N"):
            return wordnet.NOUN
        elif tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def remove_stopwords(self, text):
        """function to remove the stop words from the text"""
        words = [
            word
            for word in text.split()
            if word.isalpha() and word not in self.stop_words
        ]
        return " ".join(words)

    def lemmatize_text(self, text):
        """function to derive the base lemma form of the text using the pos tag"""
        # Get position tags
        word_pos_tags = nltk.pos_tag(word_tokenize(self.remove_stopwords(text)))
        # Map the position tag and lemmatize the word/token
        words = [
            self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(tag[1]))
            for _, tag in enumerate(word_pos_tags)
        ]
        return " ".join(words)


nltk_setup()