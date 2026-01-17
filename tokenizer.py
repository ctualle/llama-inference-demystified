#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:26:16 2024

@author: ctualle
"""

# Special character used to indicate word boundaries in tokenization
SPECIAL_CHARACTER = '▁'


# Function to add a bigram (pair of adjacent characters) to the work list
def addBigram(jprev, jnext, Symbols, Work, Key):
    """
    Add a bigram to the work list if the concatenation of two symbols 
    (previous and next) exists in the token vocabulary.
    
    Parameters:
        jprev (int): Index of the previous symbol.
        jnext (int): Index of the next symbol.
        Symbols (list of lists): List of symbols, where each symbol is a list containing the character and its indices (previous, next).
        Work (list): List of bigrams and their corresponding scores.
        Key (dict): Dictionary containing the tokenizer tokens and scores.
    
    Returns:
        None : The function modifies the Work list in place by appending valid bigrams.
    """
    txt1 = Symbols[jprev][0]  # Retrieve the previous symbol
    txt2 = Symbols[jnext][0]  # Retrieve the next symbol
    
    if (txt1 == '') or (txt2 == ''):
        return  # Skip if either of the symbols is empty
    
    txt = txt1 + txt2  # Concatenate the two symbols
    
    try:
        # Check if the concatenated bigram exists in the token vocabulary
        ind = Key["tokenizer.ggml.tokens"].index(txt)
    except:
        return  # Exit if the bigram is not found
    
    # Append the score and indices of the bigram to the Work list
    Work.append([Key["tokenizer.ggml.scores"][ind], jprev, jnext, txt])
    
    return


# Function to tokenize a given input text
def tokenize(texte, Key):
    """
    Tokenize the input text using a bigram-based approach and the provided tokenizer dictionary.
    
    Parameters:
        texte (str): The input text to be tokenized.
        Key (dict): Dictionary containing the tokenizer tokens and scores.
    
    Returns:
        list of int : List of token indices representing the input text.
    """
    # Split the text into words and add a special character as a prefix to each word
    L = texte.split()
    Txt = ''
    for k in L:
        Txt += SPECIAL_CHARACTER + k  # Add the special character before each word
    
    # Initialize a list of symbols, where each character is stored with its neighbors' indices
    Symbols = []
    TxtLength = len(Txt)  # Get the length of the entire text
    for i in range(TxtLength):
        Symbols.append([Txt[i], i - 1, i + 1])  # Store the character with its previous and next index
    
    # Initialize an empty list to store bigrams to be processed
    Work = []
    
    # Iterate over the text to add all initial bigrams to the Work list
    for j in range(len(Txt) - 1):
        addBigram(j, j + 1, Symbols, Work, Key)
    
    # Process bigrams until no more bigrams are left in the Work list
    while len(Work) > 0:
        # Find the bigram with the maximum score in the Work list
        max_value = max(Work, key=lambda item: item[0])
        max_ind = Work.index(max_value)
        Work.pop(max_ind)  # Remove the bigram with the maximum score from the list
        
        jmin = max_value[1]  # Start of the bigram (first character index)
        jmax = max_value[2]  # End of the bigram (second character index)
        txt = max_value[3]   # The bigram text (concatenated characters)
        txt1 = Symbols[jmin][0]  # Retrieve the first character of the bigram
        txt2 = Symbols[jmax][0]  # Retrieve the second character of the bigram
        
        if txt == txt1 + txt2:  # Check if the bigram is correctly formed
            # Merge the two symbols by updating the first symbol and marking the second as empty
            Symbols[jmin][0] = txt1 + txt2
            Symbols[jmax][0] = ''
            new_jmax = Symbols[jmax][2]  # Update the next index for the merged symbol
            Symbols[jmin][2] = new_jmax  # Set the next index of the first character to skip the merged symbol
            
            # If the new previous index exists, add the bigram formed with the new previous symbol
            njmin = Symbols[jmin][1]
            if njmin >= 0:
                addBigram(njmin, jmin, Symbols, Work, Key)
            
            # If the new next index exists, add the bigram formed with the new next symbol
            if new_jmax < TxtLength:
                Symbols[new_jmax][1] = jmin
                addBigram(jmin, new_jmax, Symbols, Work, Key)
    
    # Initialize an empty list to store the final token indices
    Token = []
    
    # Convert the remaining symbols to token indices
    for k in Symbols:
        txt = k[0]
        if txt != '':  # Skip empty symbols
            Token.append(Key["tokenizer.ggml.tokens"].index(txt))
    
    # Add predefined tokens at the beginning and end of the token list (optional)
    Token = [529, 29989, 326, 29918, 2962, 29989, 29958, 1792, 13] + Token + [13, 29966, 29989, 326, 29918, 355, 29989, 29958, 13, 29966, 29989, 326, 29918, 2962, 29989, 29958, 465, 22137, 13]
    
    return Token
