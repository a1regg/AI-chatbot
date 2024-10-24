import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from typing import List, Dict, Any

class Chatbot:
    def __init__(self) -> None:
        """
        Initializes the Chatbot instance, sets up the lemmatizer, 
        and loads the necessary data for the chatbot to function.
        """
        self.lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
        self.load_data()

    def load_data(self) -> None:
        """
        Loads intents from a JSON file, as well as the words and classes 
        from pickle files. Also loads the pre-trained model for predictions.
        """
        # Load intents
        with open('data/intents.json', 'r', encoding='utf-8') as file:
            self.intents: Dict[str, Any] = json.load(file)

        self.words: list[str] = pickle.load(open('data/words.pkl', 'rb'))
        self.classes: list[str] = pickle.load(open('data/classes.pkl', 'rb'))
        self.model = load_model('data/chatbot_model.keras')

    def clean_up_sentence(self, sentence: str) -> list[str]:
        """
        Cleans up the input sentence by tokenizing and lemmatizing the words.

        Args:
            sentence (str): The input sentence to be cleaned.

        Returns:
            list[str]: A list of lemmatized words from the input sentence.
        """
        sentence_words: list[str] = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence: str) -> np.ndarray:
        """
        Converts the input sentence into a bag-of-words representation.

        Args:
            sentence (str): The input sentence to be converted.

        Returns:
            np.ndarray: A numpy array representing the bag-of-words.
        """
        sentence_words: list[str] = self.clean_up_sentence(sentence)
        bag: list[int] = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence: str) -> list[Dict[str, Any]]:
        """
        Predicts the class of the input sentence using the trained model.

        Args:
            sentence (str): The input sentence for which to predict the class.

        Returns:
            list[Dict[str, Any]]: A list of dictionaries containing the predicted 
            intent and its associated probability.
        """
        bag: np.ndarray = self.bag_of_words(sentence)
        res: np.ndarray = self.model.predict(np.array([bag]))[0]
        ERROR_THRESHOLD: float = 0.25
        
        results: list[list[Any]] = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return_list: list[Dict[str, str]] = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        
        return return_list

    def get_response(self, intents_list: list[Dict[str, Any]], lang_prefix: str) -> str:
        """
        Retrieves a response based on the predicted intent.

        Args:
            intents_list (list[Dict[str, Any]]): A list of predicted intents.
            lang_prefix (str): The language prefix to select the appropriate response.

        Returns:
            str: A response string based on the predicted intent.
        """
        tag: str = intents_list[0]['intent']
        list_of_intents: list[Dict[str, Any]] = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                responses: list[str] = i['responses'][lang_prefix]
                return random.choice(responses)
        return "I'm sorry, I didn't understand that."