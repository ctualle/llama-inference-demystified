#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:11:20 2024

@author: ctualle
"""

import tkinter as tk
import Loader
import tokenizer
import transformer
import decompression
import check
import numpy as np
import pickle

# Path to adapt 
model_llama = '../models/llama-2-7b-chat.Q8_0/'

# On-click handler
def print_text():
    #try:
        [Keys,Tensors] = Loader.load_model(model_llama)
        user_input = entree.get()  # Get the text from the input
        tkn = [1,2]
        Context = [ [] , [] ]
        NewToken = transformer.llama_model(tkn,Context,Keys,Tensors)

# main frame
window = tk.Tk()
window.title("Local Chatbot")

# Input field
entry = tk.Entry(window, width=50)
entry.pack(pady=10)  

# Button to submit the text
send_button = tk.Button(window, text="Send", command=print_text)
send_button.pack(pady=10)

# Display label
label_text = tk.Label(window, text="", font=("Helvetica", 14))
label_text.pack(pady=10)

# Start
window.mainloop()
