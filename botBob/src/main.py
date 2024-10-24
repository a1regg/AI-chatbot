import sys
from PyQt5 import QtWidgets
from gui import ChatbotGUI, LanguageSelectionDialog  

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    lang_dialog = LanguageSelectionDialog()
    if lang_dialog.exec_() == QtWidgets.QDialog.Accepted:
        selected_language = lang_dialog.get_selected_language()
        gui = ChatbotGUI(selected_language)
        gui.show()
        sys.exit(app.exec_())
