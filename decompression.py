#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:05:33 2024

@author: ctualle
"""
import struct
import numpy as np
import Loader

def fp16_to_fp32(fp16_as_int): #put fp16 in integer
    
    # Extract sign, exponent, and fraction from fp16
    sign = (fp16_as_int & 0x8000) >> 15
    exponent = (fp16_as_int & 0x7C00) >> 10
    fraction = fp16_as_int & 0x03FF

    # Compute fp32 components
    if exponent == 0:  
        if fraction == 0:
            # Zero case
            fp32_exponent = 0
            fp32_fraction = 0
        else:
            # Subnormal case
            exponent_shift = 10 - fraction.bit_length()
            fp32_exponent = 127 - 15 - exponent_shift + 1
            fp32_fraction = fraction << (23 - 10 + exponent_shift)
    elif exponent == 0x1F:
        # Infinity or NaN case
        fp32_exponent = 255
        fp32_fraction = fraction << (23 - 10)
    else:
        # Normal case
        fp32_exponent = exponent + (127 - 15)
        fp32_fraction = fraction << (23 - 10)

    # Reassemble fp32
    fp32_as_int = (sign << 31) | (fp32_exponent << 23) | fp32_fraction
    
    # Pack as bytes and unpack as float
    packed_fp32 = struct.pack('I', fp32_as_int)
    fp32 = struct.unpack('f', packed_fp32)[0]

    return fp32

FP16_TO_FP32 = [fp16_to_fp32(i) for i in range(65536)]

def dequantize_q8_0 (buf,size, bloc_size, raw_size, typestruct):
    y = np.empty(size, dtype = 'float32')
    cnt_y = 0
    for cnt_buf in range(0,len(buf),raw_size):
        [d16, s] = struct.unpack( typestruct,buf[cnt_buf:cnt_buf+raw_size] )
        i_cnt_y = cnt_y + bloc_size
        y[cnt_y:i_cnt_y] = FP16_TO_FP32[d16]*np.array(list(s)).astype('int8')
        cnt_y = i_cnt_y
    return y

    
def token_embedding(Model,tokens,K,T):
    nb_tokens = len(tokens)   # number of tokens to treat
    #N_EMBD = Model.Key['llama.embedding_length']
    N_EMBD = K['llama.embedding_length']
    #W_embd_T = Model.TensorC['token_embd.weight']
    W_embd_T = T['token_embd.weight']
    W_type = W_embd_T['tensor_type']
    W_embd = W_embd_T['data']
    [bloc_size, raw_size, typestruct] = Model.gguf_tensor_type[W_type]
    N_EMBD_q8 = (N_EMBD * raw_size) // bloc_size                 
 
    embd_token = np.empty((nb_tokens, N_EMBD), dtype = 'float32')
    for i in range(nb_tokens):
        cur = tokens[i]*N_EMBD_q8
        buf = W_embd[cur:cur+N_EMBD_q8] 
        embd_token[i,:]=dequantize_q8_0(buf,N_EMBD,bloc_size, raw_size, typestruct)
        
    return embd_token    
    

def extract(blkTensor):
    Ttype = blkTensor['tensor_type'] 
    Tsize = blkTensor['taille_dimensions'][::-1]
    N = np.prod(Tsize)
    
    if (Ttype=='Q8_0'):
        [bloc_size, raw_size, typestruct] = [32,34,'<H32s']
        data = dequantize_q8_0 (blkTensor['data'], N , bloc_size, raw_size, typestruct)
    elif (Ttype=='F32'):
        data = np.array(struct.unpack('<'+str(N)+'f', blkTensor['data']),dtype = 'float32')
    data = data.reshape(Tsize)
    return data
        
    
