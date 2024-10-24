import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from botBob import Chatbot

class LanguageSelectionDialog(QtWidgets.QDialog):
    """Dialog for selecting the language for the chatbot."""

    def __init__(self) -> None:
        """Initialize the language selection dialog."""
        super().__init__()
        self.setWindowTitle("Select Language")
        self.setGeometry(100, 100, 300, 150)

        self.layout = QtWidgets.QVBoxLayout()

        self.language_label = QtWidgets.QLabel("Select Language:")
        self.layout.addWidget(self.language_label)

        self.language_combo = QtWidgets.QComboBox(self)
        self.language_combo.addItems(["English (en)", "Ukrainian (uk)"])  
        self.layout.addWidget(self.language_combo)

        self.confirm_button = QtWidgets.QPushButton("Confirm", self)
        self.confirm_button.clicked.connect(self.accept)
        self.layout.addWidget(self.confirm_button)

        self.setLayout(self.layout)

    def get_selected_language(self) -> str:
        """Get the selected language code.

        Returns:
            str: The language code corresponding to the selected language.
        """
        language_map: dict[str, str] = {
            "English (en)": "en",
            "Ukrainian (uk)": "uk"
        }
        return language_map[self.language_combo.currentText()]

class ChatbotGUI(QtWidgets.QWidget):
    """Main GUI for the chatbot."""

    def __init__(self, lang_prefix: str) -> None:
        """Initialize the chatbot GUI.

        Args:
            lang_prefix (str): The language prefix for the chatbot responses.
        """
        super().__init__()
        self.chatbot = Chatbot()
        self.lang_prefix = lang_prefix
        self.initUI()

    def initUI(self) -> None:
        """Set up the user interface components."""
        self.setWindowTitle('Bob')
        self.setGeometry(100, 100, 400, 500)

        # Layout
        self.layout = QtWidgets.QVBoxLayout()

        # Theme selection
        self.theme_label = QtWidgets.QLabel("Select Theme:")
        self.layout.addWidget(self.theme_label)

        self.theme_combo = QtWidgets.QComboBox(self)
        self.theme_combo.addItems(["Light", "Dark"])
        self.theme_combo.currentIndexChanged.connect(self.change_theme)
        self.layout.addWidget(self.theme_combo)

        # Chat window
        self.chat_window = QtWidgets.QTextBrowser(self)
        self.layout.addWidget(self.chat_window)

        # User input
        self.user_input = QtWidgets.QLineEdit(self)
        self.layout.addWidget(self.user_input)

        # Send button
        self.send_button = QtWidgets.QPushButton('Send', self)
        self.send_button.clicked.connect(self.send_message)
        self.layout.addWidget(self.send_button)

        self.setLayout(self.layout)

    def change_theme(self) -> None:
        """Change the theme of the GUI based on user selection."""
        theme: str = self.theme_combo.currentText()
        if theme == "Dark":
            self.setStyleSheet("background-color: #2E2E2E; color: white;")
            self.chat_window.setStyleSheet("background-color: #2E2E2E; color: white;")
            self.user_input.setStyleSheet("background-color: #3E3E3E; color: white;")
        else:
            self.setStyleSheet("background-color: white; color: black;")
            self.chat_window.setStyleSheet("background-color: white; color: black;")
            self.user_input.setStyleSheet("background-color: white; color: black;")

    def send_message(self) -> None:
        """Send the user's message to the chatbot and display the response."""
        user_input: str = self.user_input.text()
        if user_input:
            self.chat_window.append(f"You: {user_input}")
            self.user_input.clear()

            intents = self.chatbot.predict_class(user_input)
            if intents:
                response: str = self.chatbot.get_response(intents, self.lang_prefix)
                self.chat_window.append(f"Bob: {response}")
            else:
                self.chat_window.append("Bob: I'm sorry, I didn't understand that.")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # Show language selection dialog
    lang_dialog = LanguageSelectionDialog()
    if lang_dialog.exec_() == QtWidgets.QDialog.Accepted:
        selected_language: str = lang_dialog.get_selected_language()
        gui = ChatbotGUI(selected_language)
        gui.show()
        sys.exit(app.exec_())
