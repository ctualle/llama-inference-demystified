#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:59:56 2024

@author: ctualle
"""

import struct
import pickle
    
DEFAULT_ALIGNMENT = 32


def load_model(rep):
    with open(rep+'Keys.pkl','rb') as f:
        Keys = pickle.load(f)
    with open(rep+'Tensors_List.pkl','rb') as f:
        List = pickle.load(f)
    Tensors = {}    
    for tensor_name in List:
        with open(rep+tensor_name+'.pkl','rb') as f:
            Tensors[tensor_name] = pickle.load(f)
            print(tensor_name + ' loaded')
    return Keys,Tensors        

class Model:
    # Dictionnaire pour la correspondance des types gguf avec struct
    #  "type": [ symbole pour conversion Python, type size ]
    gguf_type = { 
        "UINT8":  ["B", 1], 
        "INT8":   ["b", 1],
        "UINT16": ["H", 2],
        "INT16":  ["h", 2],
        "UINT32": ["I", 4],
        "INT32":  ["i", 4],
        "FLOAT32":["f", 4],
        "BOOL":   ["?", 1],
        "STR":    ["Error", None],
        "ARRAY":  ["Error", None],  
        "UINT64": ["Q", 8],
        "INT64":  ["q", 8],
        "FLOAT64":["d", 8],
    }
    
    gguf_tensor_type = {
      "F32": [ 1, 4,'<f'],      #"type":["block_size","type_size"]
      "F16": [1,2,'<H'],
      "Q4_0": [None,None,None],
      "Q4_1": [None,None,None],
      "Q4_2": [None,None,None],
      "Q4_3": [None,None,None],
      "Q5_0": [None,None,None],
      "Q5_1": [None,None,None],
      "Q8_0": [32,34,'<H32s'],
      "Q8_1": [None,None,None],
      "Q2_K": [None,None,None],
      "Q3_K": [None,None,None],
      "Q4_K": [None,None,None],
      "Q5_K": [None,None,None],
      "Q6_1": [None,None,None],
      "Q8_K": [None,None,None],
      "IQ2_XXS": [None,None,None],
      "IQ2_XS": [None,None,None],
      "IQ3_XS": [None,None,None],
      "IQ1_S": [None,None,None],
      "IQ4_NL": [None,None,None],
      "IQ3_S": [None,None,None],
      "IQ2_S": [None,None,None],
      "IQ4_XS": [None,None,None],
      "I8": [None,None,None],
      "IQ16": [None,None,None],
      "I32": [None,None,None],
      "I64": [None,None,None],
      "F64": [None,None,None],
      "IQ1_M": [None,None,None],
      "BF16": [None,None,None],
        }

    # Liste des types pour indexation
    type_list = list(gguf_type.keys())
    tensortype_list = list(gguf_tensor_type.keys())

    def __init__(self):
        self.Version = 0
        self.n_tensors = 0
        self.n_kv = 0
        self.Key = {}
        self.TensorC = {}

    def read_gguftype(self, fichier, type_name):
        """Lit une valeur du type spécifié depuis le fichier."""
        if type_name == "STR":
            str_size = self.read_gguftype(fichier, "INT64")
            value = fichier.read(str_size).decode('utf-8')
        elif type_name == "ARRAY":
            tab_typename = self.type_list[self.read_gguftype(fichier, "INT32")]
            tab_N = self.read_gguftype(fichier, "INT64")
            value = [self.read_gguftype(fichier, tab_typename) for _ in range(tab_N)]
        else:
            size = self.gguf_type[type_name][1]
            bloc = fichier.read(size)
            value = struct.unpack('<' + self.gguf_type[type_name][0], bloc)[0]
        return value
    
    def Alignment(self,fichier):
        try:
            align = self.Key['general.alignment']
        except:
            align = DEFAULT_ALIGNMENT
        position = fichier.tell()
        a = align - position%align
        fichier.read(a)
        
        
        

    def MyLoad(self, nom):
        """Charge les données depuis le fichier binaire spécifié."""
        with open(nom, 'rb') as fichier:
            
            # Lire le contenu de l'en-tête
            b = fichier.read(4)
            self.Version = self.read_gguftype(fichier, "INT32")
            self.n_tensors = self.read_gguftype(fichier, "INT64")
            self.n_kv = self.read_gguftype(fichier, "INT64")

            # Stocker les clés et leurs valeurs dans le dictionnaire Key
            for i in range(self.n_kv):
                kv_name = self.read_gguftype(fichier, "STR")
                kv_typename = self.type_list[self.read_gguftype(fichier, "INT32")]
                value = self.read_gguftype(fichier, kv_typename)
                self.Key[kv_name] = value

            # Stocker les informations des tenseurs dans le dictionnaire TensorC
            for i in range(self.n_tensors):
                tensor_name = self.read_gguftype(fichier, "STR")
                dimension = self.read_gguftype(fichier, "INT32")
                tdim = [self.read_gguftype(fichier, "INT64") for _ in range(dimension)]
                tensor_type = self.tensortype_list[self.read_gguftype(fichier, "INT32")]
                offset = self.read_gguftype(fichier, "INT64")
                self.TensorC[tensor_name] = {'dimension': dimension, 
                                             'taille_dimensions': tdim, 
                                             'tensor_type': tensor_type,
                                             'offset': offset,
                                             'data': [] }

            # Loader les tenseurs
            Tlist = list(self.TensorC.keys())
            Toffset = [self.TensorC[nm]['offset'] for nm in Tlist]

            dictn = self.Key
            self.Alignment(fichier)
            print(fichier.tell())
            for i in range(self.n_tensors-1):
                tensor_name = Tlist[i]
                Tsize = Toffset[i+1]-Toffset[i]
                self.TensorC[tensor_name]['data'] = fichier.read(Tsize)
            # On lit jusqu'à la fin
            self.TensorC[Tlist[self.n_tensors-1]]['data'] = fichier.read()

# Exemple d'utilisation :
# model = Model()
# model.MyLoad('nom_du_fichier.bin')
# print(model.Key)
# print(model.TensorC)
