from Orange.widgets import gui, widget, settings
from Orange.widgets.widget import OWWidget, Input, Output
from AnyQt.QtWidgets import QTextEdit, QPushButton, QLineEdit, QVBoxLayout
from AnyQt.QtCore import Qt

import google.generativeai as genai


class OWLLMChatbot(OWWidget):
    name = "LLM Chatbot"
    description = "A chatbot widget using Google's Generative AI"
    icon = "icons/ow_llm.svg"
    priority = 10

    class Inputs:
        data = Input("Data", object)

    class Outputs:
        pass

    want_main_area = True

    # Widget parameters
    api_key = settings.Setting("")
    system_prompt = settings.Setting("""
    You are an AI assistant specialized in the Orange Data Mining application. 
    Your role is to help students understand and use Orange for data analysis, 
    visualization, and machine learning tasks. Provide detailed, accurate answers 
    to questions about Orange. Use code examples or workflow descriptions when appropriate.
    """)

    def __init__(self):
        super().__init__()

        # GUI
        box = gui.widgetBox(self.controlArea, "Configuration")

        # API Key input
        self.api_key_input = gui.lineEdit(box, self, "api_key", "API Key:")

        # System Prompt input
        self.system_prompt_input = QTextEdit(self.controlArea)
        self.system_prompt_input.setPlainText(self.system_prompt)
        self.system_prompt_input.textChanged.connect(self.system_prompt_changed)
        box.layout().addWidget(self.system_prompt_input)

        # Initialize button
        self.init_button = gui.button(box, self, "Initialize Chatbot", callback=self.initialize_chat)

        # Chat display
        self.chat_display = QTextEdit(self.mainArea)
        self.chat_display.setReadOnly(True)
        self.mainArea.layout().addWidget(self.chat_display)

        # User input
        self.user_input = QLineEdit(self.mainArea)
        self.mainArea.layout().addWidget(self.user_input)

        # Send button
        self.send_button = QPushButton("Send", self.mainArea)
        self.send_button.clicked.connect(self.send_message)
        self.mainArea.layout().addWidget(self.send_button)

        # Initialize variables
        self.model = None
        self.chat = None

    def system_prompt_changed(self):
        self.system_prompt = self.system_prompt_input.toPlainText()
        if self.chat:
            self.chat.send_message(self.system_prompt)

    def initialize_chat(self):
        self.api_key = self.api_key_input.text()
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.chat = self.model.start_chat(history=[])
                self.chat.send_message(self.system_prompt)
                self.chat_display.clear()
                self.chat_display.append("Chatbot initialized. You can start asking questions.")
            except Exception as e:
                self.chat_display.clear()
                self.chat_display.append(f"Error initializing chatbot: {str(e)}")
        else:
            self.chat_display.clear()
            self.chat_display.append("Please enter your API key and click 'Initialize Chatbot'.")

    def send_message(self):
        if not self.chat:
            self.chat_display.append("Please initialize the chatbot first.")
            return

        user_message = self.user_input.text()
        if user_message:
            self.chat_display.append(f"You: {user_message}")
            try:
                response = self.chat.send_message(user_message)
                self.chat_display.append(f"AI: {response.text}")
            except Exception as e:
                self.chat_display.append(f"Error: {str(e)}")
            self.user_input.clear()

    @Inputs.data
    def set_data(self, data):
        if data is not None and self.chat:
            self.chat.send_message(
                f"The current dataset has {len(data)} instances and {len(data.domain.attributes)} features.")


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWLLMChatbot).run()