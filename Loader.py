#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul 18 16:59:56 2024
This script is designed to read and process binary data, specifically for
loading tensors and key-value pairs from a binary file format. The script
includes functionality to handle different data types, tensor formats,
alignment issues, and decompression of tensor data using a custom module.

@author: ctualle
"""

import struct
import decompression
import pickle
import subprocess
import sys

try:
  import requests
except:
  subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
  import requests

try:  
  import os
except:
  subprocess.check_call([sys.executable, "-m", "pip", "install", "os"])
  import os

try:  
  from tqdm import tqdm
except:
  subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
  from tqdm import tqdm


# Default alignment used when reading from the binary file
DEFAULT_ALIGNMENT = 32

# Dictionary mapping data types to their struct format and size in bytes
gguf_type = {
    "UINT8": ["B", 1], 
    "INT8": ["b", 1], 
    "UINT16": ["H", 2], 
    "INT16": ["h", 2],
    "UINT32": ["I", 4],
    "INT32": ["i", 4],
    "FLOAT32":["f", 4],
    "BOOL": ["?", 1],
    "STR": ["Error", None], 
    "ARRAY": ["Error", None],
    "UINT64": ["Q", 8],
    "INT64": ["q", 8],
    "FLOAT64":["d", 8],
}

# Dictionary mapping tensor data types to their attributes: block size, type size, and struct format
gguf_tensor_type = {
    "F32": [1, 4, '<f'],
    "F16": [1, 2, '<H'], 
    
    # Quantized tensor types (Q) - details unknown
    "Q4_0": [None, None, None],
    "Q4_1": [None, None, None],
    "Q4_2": [None, None, None],
    "Q4_3": [None, None, None],
    "Q5_0": [None, None, None],
    "Q5_1": [None, None, None],
    "Q8_0": [32, 34, '<H32s'], 
    
    # More tensor types and formats, details not provided
    "Q8_1": [None, None, None],
    "Q2_K": [None, None, None],
    "Q3_K": [None, None, None],
    "Q4_K": [None, None, None],
    "Q5_K": [None, None, None],
    "Q6_1": [None, None, None],
    "Q8_K": [None, None, None],
    "IQ2_XXS": [None, None, None],
    "IQ2_XS": [None, None, None],
    "IQ3_XS": [None, None, None],
    "IQ1_S": [None, None, None],
    "IQ4_NL": [None, None, None],
    "IQ3_S": [None, None, None],
    "IQ2_S": [None, None, None],
    "IQ4_XS": [None, None, None],
    "I8": [None, None, None],
    "IQ16": [None, None, None],
    "I32": [None, None, None],
    "I64": [None, None, None],
    "F64": [None, None, None],
    "IQ1_M": [None, None, None],
    "BF16": [None, None, None],
}

# Lists of types for indexing
type_list = list(gguf_type.keys())
tensortype_list = list(gguf_tensor_type.keys())

# Function to read a value of a specific type from the file
def read_gguftype(fichier, type_name):
    """Reads a value of the specified type from the file."""
    if type_name == "STR":
        # For strings, read the length first (as INT64), then read the string data
        str_size = read_gguftype(fichier, "INT64")
        value = fichier.read(str_size).decode('utf-8')
    elif type_name == "ARRAY":
        # For arrays, read the type and size of the array, then the array elements
        tab_typename = type_list[read_gguftype(fichier, "INT32")]
        tab_N = read_gguftype(fichier, "INT64")
        value = [read_gguftype(fichier, tab_typename) for _ in range(tab_N)]
    else:
        # For all other types, use the struct format and size defined in gguf_type
        size = gguf_type[type_name][1]
        bloc = fichier.read(size)
        value = struct.unpack('<' + gguf_type[type_name][0], bloc)[0]
    return value

# Function to align the file position to the next alignment boundary
def Alignment(fichier, Key):
    """Aligns the file position based on the alignment value in the Key dictionary."""
    try:
        align = Key['general.alignment'] # Get alignment from the Key dictionary
    except KeyError:
        align = DEFAULT_ALIGNMENT # Use default alignment if not specified
    position = fichier.tell() # Current file position
    a = align - position % align # Calculate padding needed to align
    fichier.read(a) # Move the file pointer to the next aligned position

def installModel(rep):
    
    os.makedirs(rep, exist_ok=True)
    os.chdir(rep)

    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf?download=true"
    output_file = "llama-2-7b-chat.Q8_0.gguf"
    print("Loading ",output_file," on huggingface.co...")

    # streaming=True pour ne pas tout charger en mémoire
    with requests.get(url, stream=True) as r:
      r.raise_for_status()
      total_size = int(r.headers.get('content-length', 0))  # taille totale en octets
      chunk_size = 1024  # taille du bloc (1 KB)
    
      # tqdm pour la barre de progression
      with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_file) as pbar:
         with open(output_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
      
    Model_save(output_file)


def load_model(rep):
   
    """Load the keys (Keys) from a pickle file located in the directory 'rep', which is is the path where the model files are stored"""
    try:
     with open(rep+'Keys.pkl', 'rb') as f:
       Keys = pickle.load(f) # Load the dictionary of keys from the 'Keys.pkl' file
    except:
       print('Model not installed; do you want to install it (y/n) ?')
       i = input()
       if (i=='y') or (i=='Y'):
          try:
            installModel(rep)
          except:
            print("Install error ...")
            exit(1)
          print("Model correctly installes: please restart the program ...")  
       else:
          exit(1)
       
    # Load the list of tensor names (Tensors) from a pickle file
    with open(rep+'Tensors_List.pkl', 'rb') as f:
        List = pickle.load(f) # Load the list of tensor names from 'Tensors_List.pkl'

    Tensors = {}

    for tensor_name in List:
        with open(rep + tensor_name + '.pkl', 'rb') as f:
            Tensors[tensor_name] = pickle.load(f)
            print(tensor_name + ' loaded')
            
    return Keys, Tensors


def MyLoad(nom):
    """Load data from the specified GGUF file."""
    print("Model decompression ...")
    
    class Model:
        def __init__(self):
            self.Version = 0
            self.n_tensors = 0
            self.n_kv = 0
            self.Key = {}
            self.TensorC = {}
    model = Model()

    with open(nom, 'rb') as fichier:
    
     # Read the header
     b = fichier.read(4)
     model.Version = read_gguftype(fichier, "INT32")
     model.n_tensors = read_gguftype(fichier, "INT64")
     model.n_kv = read_gguftype(fichier, "INT64")
         
     # Load keys and store them in the Key dictionary
     for i in range(model.n_kv):
         kv_name = read_gguftype(fichier, "STR")
         kv_typename = type_list[read_gguftype(fichier, "INT32")]
         value = read_gguftype(fichier, kv_typename)
         model.Key[kv_name] = value
		
     # Load Tensors information and store it in the TensorC dictionary
     for i in range(model.n_tensors):
         tensor_name = read_gguftype(fichier, "STR")
         dimension = read_gguftype(fichier, "INT32")
         tdim = [read_gguftype(fichier, "INT64") for _ in range(dimension)]
         tensor_type = tensortype_list[read_gguftype(fichier, "INT32")]
         offset = read_gguftype(fichier, "INT64")
         model.TensorC[tensor_name] = {'dimension': dimension,
                                       'taille_dimensions': tdim,
                                       'tensor_type': tensor_type,
                                       'offset': offset,
                                       'data': [] }

     # Load tensors
     Tlist = list(model.TensorC.keys())
     Toffset = [model.TensorC[nm]['offset'] for nm in Tlist]
    
     dictn = model.Key
     Alignment(fichier,model.Key)
     for i in range(model.n_tensors-1):
        tensor_name = Tlist[i]
        Tsize = Toffset[i+1]-Toffset[i]
        model.TensorC[tensor_name]['data'] = fichier.read(Tsize)
     # Read untill the end
     model.TensorC[Tlist[model.n_tensors-1]]['data'] = fichier.read()

    return model

# Path to adjust
model_llama = 'llama-2-7b-chat.Q8_0.gguf'
rep = '' # Directory where the model components will be stored

def Model_save(Model_file):
    """
    This function loads the LLaMA model, extracts the tensors and keys,
    and saves them into pickle files for later use.
    """
    
    model = MyLoad(Model_file)
   
    # Model's keys and compressed tensors
    K = model.Key 
    T = model.TensorC
    
    # List to store the tensor names
    Tnames = []
    
    # Saving decompressed tensors
    for cle in T:
        print(cle)
        Tnames.append(cle)

        Tout = decompression.extract(T[cle])

        with open(rep + cle + '.pkl', 'wb') as f:
            pickle.dump(Tout, f)

    # Save the model keys (hyperparameters) to 'Keys.pkl'
    with open(rep + 'Keys.pkl', 'wb') as f:
         pickle.dump(K, f)
    print("K saved")

    # Save the list of tensor names to 'Tensors_List.pkl'
    with open(rep + 'Tensors_List.pkl', 'wb') as f:
        pickle.dump(Tnames, f)
    print("Names saved") # Confirm that tensor names are saved

    return
