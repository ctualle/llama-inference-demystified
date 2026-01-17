#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 18:39:42 2024

@author: ctualle
"""

from qtpy.QtWidgets import (QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton)
from qtpy.QtGui import QTextCursor, QPalette, QColor, QFont
from qtpy.QtCore import Qt
import Loader
import tokenizer
import transformer
import time
import decompression
import numpy as np


# Put model path
model_llama_rep = '../models/llama-2-7b-chat.Q8_0/'
model_llama_file = '../models/llama-2-7b-chat.Q8_0.gguf'


class ChatBotUI(QWidget):
    
    Complete = []     # Complete conversation history
    Context = []      # Context of the conversation
    stopSpeak = 0     # Event to stop the Chatbot
    Keys = []         # Keys used by the model
    Tensors = []     # Tensors from the model
    
    def __init__(self, *args, **kwargs):
        """
        Initializes the chatbot UI and loads the model.
        """
        # Load the model keys and tensors from the directory
        [self.Keys, self.Tensors] = Loader.load_model(model_llama_rep)
        
        super().__init__(*args, **kwargs)

        # Window
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Chat')

        # Set theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2E2E2E;
                color: white;
            }
            QTextEdit {
                background-color: #333333;
                color: black;
                border: none;
                font-family: Arial, Helvetica, sans-serif;
                font-size: 14px;
                font-weight: bold;
                line-height: 1.6;
                margin-bottom: 10px;
                background-image: url('chatnoir1-removebg-preview.png');
                background-position: right;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
            QLineEdit {
                background-color: #4D4D4D;
                color: white;
                border: 2px solid #5A5A5A;
                padding: 10px;
                font-family: Arial, Helvetica, sans-serif;
                font-size: 14px;
            }
            QPushButton {
                background-color: #5A5A5A;
                color: white;
                border: none;
                padding: 8px 16px;
                font-family: Arial, Helvetica, sans-serif;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #7A7A7A;
            }
        """)

        # Main layout 
        self.layout = QVBoxLayout()

        # Conversation area
        self.conversation_area = QTextEdit()
        self.conversation_area.setReadOnly(True) 
        self.conversation_area.setPlainText('Raoult: Welcome to the chatbot!\n\n')
        self.layout.addWidget(self.conversation_area)
        
        # Input area
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.layout.addWidget(self.user_input)

        # Send button
        self.send_button = QPushButton('Send', self)
        self.send_button.clicked.connect(self.process_message)
        self.layout.addWidget(self.send_button)

        # Final layout for the window
        self.setLayout(self.layout)
        self.show()


def closeEvent(self, event):
    """
    Handles the window close event by performing cleanup.
    """
    del self.Context
    del self.Keys
    del self.Tensors
    self.destroy()
    event.accept()

def process_message(self):
    """
    Processes the user's message and displays the chatbot's response.
    """

    user_message = self.user_input.text().strip()
    
    if user_message:
        # Append the user's message to the conversation area
        self.conversation_area.append(f'<span style="color: white;"\
        >You: {user_message}</span><br>')
        
        # Get chatbot's response
        bot_response = self.get_bot_response(user_message)
        self.conversation_area.append(f'<span style="color: violet;">Raoult:\
        {bot_response}</span><br>')
        # Clear user input field 
        self.user_input.clear()
        # Auto-scroll the conversation area to the latest message
        self.conversation_area.moveCursor(QTextCursor.MoveOperation.End)

def stopSpeak_incr(self):
    """
    Stops the chatbot from speaking.
    """
    self.stopSpeak = 1

def get_bot_response(self, message):
    """
    Retrieves the chatbot's response by tokenizing the user's input and generating a reply.
    """
    perf_TIME0 = time.time() # Track the start time
    self.Complete.append(message) # Add the user's message to the conversation history
    
    # Tokenize the user's message using the model's keys
    tkn = tokenizer.tokenize(message, self.Keys)
    perf_TOKENS = len(tkn)
    
    # Get the dictionary of tokens from the keys
    dic = self.Keys['tokenizer.ggml.tokens']
    
    # Print the tokenized message
    print([dic[j] for j in tkn])
    
    output = ""
    
    # Define the stopping markers to terminate the response generation
    StopMark = [29966, 29989, 326]
    errMark = [29966, 29989, 529]
    errMark2 = [29966, 29989, 29966]
    lenStopMark = len(StopMark)
    zeroMark = [0] * lenStopMark
    Stop = zeroMark
    MAX_TOKENS = 200
    
    # Generate tokens until a stopping marker is reached or the maximum limit is exceeded
    while (Stop[-lenStopMark:] != StopMark) and (Stop[-lenStopMark:] != errMark) and (Stop[-lenStopMark:] != errMark2) and (perf_TOKENS < MAX_TOKENS):
        print(Stop)
        print(StopMark)
        [TKout, self.Context] = transformer.llama_model(tkn, self.Context, self.Keys, \
        self.Tensors)
        print(dic[TKout])
        perf_TOKENS += 1
        tkn = [TKout]
        
        if Stop != zeroMark:
            for m in range(lenStopMark):
                output += dic[Stop[m]]
            Stop = zeroMark
            
        if TKout == 29966:
            Stop[0] = TKout
            for k in range(1, lenStopMark):
                [TKout, self.Context] = transformer.llama_model(tkn, self.Context, self.Keys, self.Tensors)
                print(dic[TKout])
            perf_TOKENS += 1
            tkn = [TKout]
            Stop[k] = TKout
        
        else:
            if TKout > 2:
                output += dic[TKout]

    # Replace special characters in the output
    output = output.replace(tokenizer.SPECIAL_CHARACTER, " ")
    output = output.replace("<0x0A>", "\n")
    output = output.replace("<unk>", "")

    # Append the generated output to the conversation history
    self.Complete.append(output)
    
    # Write performance data to a file
    file = open("perf.txt", "a")
    for i in range(len(self.Complete)):
        file.write(self.Complete[i])
        file.write("\n")
    outtxt = '\n\n ' + str(perf_TOKENS) + ' tokens in ' + str(time.time() - perf_TIME0)\
    + ' seconds\n\n'
    file.write(outtxt)
    file.close()
    
    self.stopSpeak = 0
    
    return output 


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatBotUI()
    sys.exit(app.exec())
