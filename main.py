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

# Chemin du modèle (à adapter selon l'emplacement réel du modèle)
model_llama = '../models/llama-2-7b-chat.Q8_0/'

# Fonction appelée lorsque le bouton est cliqué
def afficher_texte():
    #try:
        
        [Keys,Tensors] = Loader.load_model(model_llama)

        user_input = entree.get()  # Récupère le texte de l'entrée
        
        tkn = [1,2]
        
        Context = [ [] , [] ]
        
        NewToken = transformer.llama_model(tkn,Context,Keys,Tensors)
            
            #model = Loader.Model()
            #model.MyLoad(model_llama)
        
            
    #         Q=Q.reshape(nEMBD,Ntkn)
    #         Q=Q.transpose()
    #        # print('Q:',Q[0,0],Q[0,1],Q[0,2],Q[1,0],Q[1,1],Q[1,2])
    #        # C = check.check(Q,"../ROPE.txt")
    #         C = check.check(A,"../vecteur_attention.txt")
    #         print('A:',A[0,0],A[0,1],A[0,2],A[1,0],A[1,1],A[1,2])
            
    #         response = f"Tokenized and RMS-normalized: {C}"
    #     except Exception as e:
    #         label_texte.config(text=f"Error processing input: {str(e)}")
    #         return
        
    #     label_texte.config(text=response)  # Affiche le texte dans le label
    
    # except Exception as e:
    #      label_texte.config(text=f"Unexpected error: {str(e)}")

# Créer la fenêtre principale
fenetre = tk.Tk()
fenetre.title("Local Chatbot")

# Créer une entrée où l'utilisateur peut écrire du texte
entree = tk.Entry(fenetre, width=50)
entree.pack(pady=10)  # Ajoute de l'espace autour de l'entrée

# Créer un bouton pour envoyer le texte
bouton_envoyer = tk.Button(fenetre, text="Envoyer", command=afficher_texte)
bouton_envoyer.pack(pady=10)

# Créer un label pour afficher le texte
label_texte = tk.Label(fenetre, text="", font=("Helvetica", 14))
label_texte.pack(pady=10)

# Lancer la boucle principale de l'interface
fenetre.mainloop()
