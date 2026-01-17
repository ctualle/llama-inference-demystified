#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:14:17 2024

@author: ctualle
"""

import numpy as np
import check

def rms_norm(vector, eps,Vnorm):
    mean_square = np.mean(vector*vector,1)
    rms_value = np.sqrt(mean_square.reshape(-1,1) + eps)
    normalized_vector = Vnorm * vector / rms_value
    return normalized_vector

def matROPE(Ntkn,npast,nDIM,thscale):
    th_scales = np.array(thscale**np.array(range(nDIM//2)),dtype='float32')
    I = npast + np.array(range(Ntkn))
    theta = th_scales.reshape(-1,1) @ I.reshape(1,-1)
    return [np.cos(theta),np.sin(theta)]

def ROPE(T,c,s):
    [nHead,nDIM,ntk] = T.shape
    for n in range(nHead):
        A = T[n,0::2,:]
        B = T[n,1::2,:]
        Rx = A*c-B*s
        Ry = A*s+B*c
        T[n,0::2,:] = Rx
        T[n,1::2,:] = Ry
    return T   

def mask(alpha,npast):
    for i in range (len(alpha)):
        alpha[i,i+npast+1:]= -np.inf
        
    return alpha
    
def softmax(alpha):
    Mx = np.max(alpha,1)
    Ea = np.exp(alpha - Mx.reshape(-1,1))
    return Ea/Ea.sum(axis=1).reshape(-1,1)

def SILU(Y):
    return np.float32(Y/(1+np.exp(-Y)))

def Attention(Q,K,V):
    [nHead,nDIM,ntk] = Q.shape
    KQ_scale = np.float32(1.0/np.sqrt(nDIM))
    A = np.zeros(Q.shape,dtype='float32')
    for n in range(nHead):
        q = Q[n,:,:]
        alpha = q.transpose() @ K[n,:,:]
        alpha = mask(alpha,0)*KQ_scale
        alpha = softmax(alpha)
        A[n,:,:] = V[n,:,:] @ alpha.transpose()
    return A


def llama_model(tkn,Context,Keys,Tensors):
    
    rms_Epsilon = Keys['llama.attention.layer_norm_rms_epsilon']
    nHead = Keys['llama.attention.head_count']
    nEMBD = Keys['llama.embedding_length']
    nLayer = Keys['llama.block_count']
    nDIM = nEMBD//nHead
    L_Rope = 10000  # not in the llama keys ?
    thscale = L_Rope ** (-2.0/nDIM)
    Ntkn = len(tkn)
    npast = 0
    
    inpATTN = Tensors['token_embd.weight'][tkn,:]
   
    
    for nl in range(nLayer):
        print("Layer "+str(nl))
    
        Attn_Norm = Tensors['blk.'+str(nl)+'.attn_norm.weight']
        FFN_Norm  = Tensors['blk.'+str(nl)+'.ffn_norm.weight'] 
        W_K = Tensors['blk.'+str(nl)+'.attn_k.weight']
        W_Q = Tensors['blk.'+str(nl)+'.attn_q.weight']
        W_V = Tensors['blk.'+str(nl)+'.attn_v.weight']
        Wo = Tensors['blk.'+str(nl)+'.attn_output.weight']
        WG = Tensors['blk.'+str(nl)+'.ffn_gate.weight']
        Wup = Tensors['blk.'+str(nl)+'.ffn_up.weight']
        Wdown = Tensors['blk.'+str(nl)+'.ffn_down.weight']
        
        
        inpA_norm = rms_norm(inpATTN,rms_Epsilon,Attn_Norm)
        cur = inpA_norm.transpose()
        K = W_K @ cur
        Q = W_Q @ cur
        V = W_V @ cur
     
        K = K.reshape(nHead,nDIM,Ntkn)
        Q = Q.reshape(nHead,nDIM,Ntkn)
        V = V.reshape(nHead,nDIM,Ntkn)
        
        [Mc , Ms] = matROPE(Ntkn,npast,nDIM,thscale)
        K = ROPE(K,Mc,Ms)
        Q = ROPE(Q,Mc,Ms)
        
        A = Attention(Q,K,V)
        A = A.reshape(nEMBD,Ntkn)
        A = Wo @ A
        A = A.transpose()
        
        inpFFN = A + inpATTN
        inpFFN_Norm = rms_norm(inpFFN,rms_Epsilon,FFN_Norm)
        cur = inpFFN_Norm.transpose()
        
        G = WG @ cur
        W = Wup @ cur
        tmp = W*SILU(G)
        out = Wdown @ tmp
    
        inpATTN = out.transpose() + inpFFN

     
    Wout = Tensors['output.weight'] 
    Out_Norm = Tensors['output_norm.weight'] 
    
    out_norm = rms_norm(inpATTN,rms_Epsilon,Out_Norm).transpose()
    print(Wout.shape,out_norm.shape)
    out = Wout @ out_norm
    print(out.shape)
    out = out[:,-1]
    
    I = np.argsort(out)[-10:];
    Best = np.concatenate((I,out[I]),axis=0)
    print(Best)
 
    #A = SILU(G).transpose()
    #A = tmp.transpose()
    A = inpATTN
    # A = tmp.transpose()
    # A = G.transpose()
    # A = out.transpose()
    # A = inpFFN
    # A = inpFFN_Norm 
    print(A.size)
    C = check.check(A,"../vecteur_ADDout.txt")
    print('Erreur = ' + str(C))
    print('A:',A[0,0],A[0,1],A[0,2],A[1,0],A[1,1],A[1,2])
       
    
    return 0   
        
