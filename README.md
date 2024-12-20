# AI Chatbot

This project is an AI-powered conversational agent designed to interact with users in multiple languages, specifically English and Ukrainian. The chatbot utilizes a machine learning model trained on a set of intents defined in a JSON file, enabling it to understand user inputs and provide appropriate responses based on the trained data.

## Project Structure

The project consists of the following main files:

- **training.py**: This script is responsible for training the machine learning model. It processes the intents, prepares the training data, builds the neural network model, and saves the trained model for later use.

- **bot.py**: This file contains the `Chatbot` class, which handles the chatbot's functionality. It loads the trained model, processes user inputs, predicts the intent of the input, and retrieves the corresponding response.

- **gui.py**: This file implements the graphical user interface (GUI) for the chatbot using PyQt5. It allows users to interact with the chatbot, select themes, and input messages.

- **main.py**: This is the main entry point of the application. It initializes the application, displays the language selection dialog, and launches the chatbot GUI.

- **data/**: This directory contains the following files:
  - **intents.json**: A JSON file that defines the intents, patterns, and responses for the chatbot. Each intent has a tag, a set of patterns for user input, and corresponding responses in different languages.
  - **words.pkl**: A pickle file that stores the unique words extracted from the intents for training the model.
  - **classes.pkl**: A pickle file that stores the unique classes (tags) extracted from the intents for training the model.
  - **chatbot_model.keras**: The saved Keras model file that contains the trained neural network.

## Capabilities

- **Multi-language Support**: The chatbot can respond in English and Ukrainian, allowing users to select their preferred language at the start of the conversation.
- **Intent Recognition**: The chatbot can recognize user intents based on predefined patterns and provide appropriate responses.
- **User Interaction**: The GUI allows users to send messages and receive responses in a conversational format.

## Limitations

- **Limited Training Data**: The chatbot is trained on a limited dataset of approximately 3500 lines in the JSON file, which may restrict its ability to understand complex queries or diverse topics.
- **Grammar Sensitivity**: The chatbot struggles with inputs that contain significant grammatical errors or typos, which may lead to misunderstandings or incorrect responses.
- **Single Language Selection**: Currently, users can only select their language at the beginning of the conversation. Future versions may allow language switching during the chat.

## Future Improvements

- **Expanded Training Data**: Future versions will include a larger and more diverse dataset to improve the chatbot's understanding and response accuracy.
- **Enhanced Language Processing**: Implementing natural language processing techniques to better handle grammatical errors and informal language.
- **Dynamic Language Switching**: Allowing users to switch languages during the conversation for a more flexible interaction experience.
- **Additional Features**: Exploring the integration of more advanced features, such as context awareness and personalized responses based on user history.

## Conclusion

This chatbot project serves as a foundational step in developing my personal skills. While it has limitations, it provides a solid base for future enhancements and improvements. Feedback and contributions are welcome as I continue to refine and expand the chatbot's capabilities.

---

Feel free to clone the repository, explore the code, and contribute to the project!
