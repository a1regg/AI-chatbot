import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from typing import List, Tuple, Dict, Any

import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Initialize the lemmatizer
lemmatizer: WordNetLemmatizer = WordNetLemmatizer()

# Load intents from a JSON file
def load_intents(file_path: str) -> dict[str, Any]:
    """Load intents from a JSON file.

    Args:
        file_path (str): The path to the intents JSON file.

    Returns:
        dict[str, Any]: The loaded intents as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Process intents to extract words, classes, and documents
def process_intents(intents: dict[str, Any]) -> Tuple[list[str], list[str], list[Tuple[list[str], str, str]]]:
    """Process intents to extract words, classes, and documents.

    Args:
        intents (dict[str, Any]): The intents dictionary.

    Returns:
        Tuple[list[str], list[str], list[Tuple[list[str], str, str]]]: 
            A tuple containing a list of words, a list of classes, and a list of documents.
    """
    words: list[str] = []
    classes: list[str] = []
    documents: list[Tuple[list[str], str, str]] = []
    ignore_letters: list[str] = ['?', '!', '.', ',', '“', '”', '‘', '’']

    for intent in intents['intents']:
        if 'patterns' in intent:
            for lang in intent['patterns']:  # Iterate through the languages
                for pattern in intent['patterns'][lang]:  # Access patterns for each language
                    word_list: list[str] = nltk.word_tokenize(pattern)
                    words.extend(word_list)
                    documents.append((word_list, intent['tag'], lang))  # Store language
                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])

    # Lemmatize and filter words
    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    return words, classes, documents

def save_data(words: list[str], classes: list[str], words_file: str, classes_file: str) -> None:
    """Save words and classes to pickle files.

    Args:
        words (list[str]): The list of words to save.
        classes (list[str]): The list of classes to save.
        words_file (str): The file path to save words.
        classes_file (str): The file path to save classes.
    """
    pickle.dump(words, open(words_file, 'wb'))
    pickle.dump(classes, open(classes_file, 'wb'))

def prepare_training_data(documents: list[Tuple[list[str], str, str]], words: list[str], classes: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare training data from documents.

    Args:
        documents (list[Tuple[list[str], str, str]]): The list of documents.
        words (list[str]): The list of words.
        classes (list[str]): The list of classes.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the training features and labels.
    """
    training: list[list[int]] = []
    output_empty: list[int] = [0] * len(classes)

    for document in documents:
        bag: list[int] = []
        word_patterns: list[str] = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row: list[int] = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append(bag + output_row)

    random.shuffle(training)
    training = np.array(training)

    train_x: np.ndarray = training[:, :len(words)]
    train_y: np.ndarray = training[:, len(words):]

    return train_x, train_y

def build_model(input_shape: int, output_shape: int) -> Sequential:
    """Build and compile the neural network model.

    Args:
        input_shape (int): The shape of the input data.
        output_shape (int): The shape of the output data.

    Returns:
        Sequential: The compiled Keras model.    """
    model: Sequential = Sequential()
    model.add(Dense(256, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))

    sgd: SGD = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

# Train the model
def train_model(model: Sequential, train_x: np.ndarray, train_y: np.ndarray, epochs: int, batch_size: int) -> None:
    """Train the Keras model.

    Args:
        model (Sequential): The Keras model to train.
        train_x (np.ndarray): The training features.
        train_y (np.ndarray): The training labels.
        epochs (int): The number of epochs to train.
        batch_size (int): The batch size for training.
    """
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

# Save the trained model
def save_model(model: Sequential, file_path: str) -> None:
    """Save the trained Keras model to a file.

    Args:
        model (Sequential): The Keras model to save.
        file_path (str): The file path to save the model.
    """
    model.save(file_path)

if __name__ == "__main__":
    # Load intents
    intents: dict[str, Any] = load_intents('data/intents.json')

    # Process intents
    words: list[str]
    classes: list[str]
    documents: list[Tuple[list[str], str, str]]
    words, classes, documents = process_intents(intents)

    # Save words and classes
    save_data(words, classes, 'data/words.pkl', 'data/classes.pkl')

    # Prepare training data
    train_x: np.ndarray
    train_y: np.ndarray
    train_x, train_y = prepare_training_data(documents, words, classes)

    # Build the model
    model: Sequential = build_model(input_shape=len(train_x[0]), output_shape=len(train_y[0]))

    # Train the model
    train_model(model, train_x, train_y, epochs=20, batch_size=5)

    # Save the trained model
    save_model(model, 'data/chatbot_model.keras')