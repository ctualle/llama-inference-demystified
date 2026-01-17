#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:11:20 2024

@author: ctualle
"""

import pickle
import Loader
import decompression


# Chemin du modèle (à adapter selon l'emplacement réel du modèle)
model_llama = '../models/llama-2-7b-chat.Q8_0.gguf'
rep = '../models/llama-2-7b-chat.Q8_0/'

def save() :
    
    model = Loader.Model()
    model.MyLoad(model_llama)
    K = model.Key
    T = model.TensorC
    Tnames =  []
    
    for cle in T:
        print(cle)
        Tnames = Tnames+[cle]
        Tout = decompression.extract(T[cle])
        with open(rep+cle+'.pkl','wb') as f:
            pickle.dump(Tout,f)

    with open(rep+'Keys.pkl','wb') as f:
        pickle.dump(K,f)
    print("K saved")

    with open(rep+'Tensors_List.pkl','wb') as f:
        pickle.dump(Tnames,f)
    print("Names saved")
        
        
    return
