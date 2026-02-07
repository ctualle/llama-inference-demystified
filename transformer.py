#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:14:17 2024

@author: ctualle
"""

import numpy as np
import check


# Function to normalize vectors using Root Mean Square (RMS)
def rms_norm(vector, eps, Vnorm):
    """
    Normalize the input vector using RMS and a normalization factor (Vnorm).
    
    Parameters:
        vector (numpy array): Input vector to be normalized.
        eps (float): Small value added to avoid division by zero.
        Vnorm (float): Normalization factor.
    
    Returns:
        numpy array : Normalized vector.
    """
    mean_square = np.mean(vector * vector, len(vector.shape) - 1)
    rms_value = np.sqrt(mean_square.reshape(-1, 1) + eps)
    normalized_vector = Vnorm * vector / rms_value
    return normalized_vector

# Function to calculate rotation matrix for Rotary Positional Encoding (ROPE)
def matROPE(Ntkn, npast, nDIM, thscale):
    """
    Compute cosine and sine values for Rotary Positional Encoding (ROPE).
    
    Parameters:
        Ntkn (int): Number of tokens.
        npast (int): Number of past tokens.
        nDIM (int): Dimension of the model.
        thscale (float): Scaling factor for the positional encoding.
    
    Returns:
        tuple of numpy arrays : Cosine and sine values for ROPE.
    """
    th_scales = np.array(thscale ** np.array(range(nDIM // 2)), dtype='float32')
    I = npast + np.array(range(Ntkn))
    theta = th_scales.reshape(-1, 1) @ I.reshape(1, -1)
    return [np.cos(theta), np.sin(theta)]

# Function to apply ROPE on query and key matrices in attention mechanism
def ROPE(T, c, s):
    """
    Apply Rotary Positional Encoding (ROPE) on input tensor T.
    
    Parameters:
        T : numpy array
            Input tensor containing queries or keys.
        c : numpy array
            Cosine values.
        s : numpy array
            Sine values.
    
    Returns:
        numpy array : Tensor after applying ROPE.
    """
    [nHead, nDIM, ntk] = T.shape
    for n in range(nHead):
        A = T[n, 0::2, :]
        B = T[n, 1::2, :]
        Rx = A * c - B * s
        Ry = A * s + B * c
        T[n, 0::2, :] = Rx
        T[n, 1::2, :] = Ry
    return T

# Function to apply attention mask (to prevent attending to future tokens)
def mask(alpha, npast):
    """
    Apply mask to prevent attention to future tokens.
    
    Parameters:
        alpha (numpy array): Attention scores.
        npast (int): Number of past tokens.
    
    Returns:
        numpy array : Masked attention scores.
    """
    for i in range(len(alpha)):
        alpha[i, i + npast + 1:] = -np.inf  # Mask future tokens
    return alpha

# Function to apply softmax operation
def softmax(alpha):
    """
    Apply softmax function to normalize attention scores.
    
    Parameters:
        alpha (numpy array): Attention scores.
    
    Returns:
        numpy array : Softmax-normalized scores.
    """
    Mx = np.max(alpha, len(alpha.shape) - 1)
    Ea = np.exp(alpha - Mx.reshape(-1, 1))  # Subtract max for numerical stability
    return Ea / Ea.sum(axis=1).reshape(-1, 1)

# Function to apply Swish activation (also known as SILU)
def SILU(Y):
    """
    Apply the Sigmoid Linear Unit (SILU) or Swish activation function.
    
    Parameters:
        Y : numpy array
            Input data.
    
    Returns:
        numpy array : Activated data.
    """
    return np.float32(Y / (1 + np.exp(-Y)))

# Function for scaled dot-product attention
def Attention(Q, K, V, npast):
    """
    Compute the attention mechanism using queries, keys, and values.
    
    Parameters:
        Q (numpy array): Query matrix.
        K (numpy array): Key matrix.
        V (numpy array): Value matrix.
        npast (int): Number of past tokens.
    
    Returns:
        numpy array : Output of the attention mechanism.
    """
    [nHead, nDIM, ntk] = Q.shape
    KQ_scale = np.float32(1.0 / np.sqrt(nDIM))  # Scale factor for attention
    A = np.zeros(Q.shape, dtype='float32')
    for n in range(nHead):
        q = Q[n, :, :]
        alpha = q.transpose() @ K[n, :, :]  # Compute attention scores
        alpha = mask(alpha, npast) * KQ_scale  # Apply mask and scaling
        alpha = softmax(alpha)  # Apply softmax to get attention weights
        A[n, :, :] = V[n, :, :] @ alpha.transpose()  # Apply attention weights to values
    return A

# Main Llama model function
def llama_model(tkn, Context, Keys, Tensors):
    """
    Implements the Llama model to generate text based on input tokens.
    
    Parameters:
        tkn (list of int): Input tokens.
        Context (numpy array): Previous context (states of past tokens).
        Keys (dict): Model hyperparameters and pre-trained data.
        Tensors (dict): Model weights and embedding information.
    
    Returns:
        tuple : Output token and updated context.
    """
   
    # Initialize constants and model hyperparameters from the Keys dictionary
    INDK = 0
    INDV = 1
    rms_Epsilon = Keys['llama.attention.layer_norm_rms_epsilon']  # Epsilon for RMS normalization
    nHead = Keys['llama.attention.head_count']  # Number of attention heads
    nEMBD = Keys['llama.embedding_length']  # Embedding size
    nLayer = Keys['llama.block_count']  # Number of layers (model blocks)
    nDIM = nEMBD // nHead  # Dimension of each attention head 
    L_Rope = 10000  # Scaling factor for Rotary Positional Encoding (ROPE)
    thscale = L_Rope ** (-2.0 / nDIM)  # Theta scaling for ROPE
    Ntkn = len(tkn)  # Number of input tokens
    
    # Initialize or update the context based on past tokens
    if len(Context) == 0: 
        # If no previous context, initialize an empty one
        npast = 0
        newContext = np.empty((2, nLayer, nHead, nDIM, Ntkn))  # Initialize an empty context array
    else:
        # If context exists, concatenate it with the new tokens
        npast = Context.shape[4]
        newContext = np.empty((2, nLayer, nHead, nDIM, Ntkn + npast))  # Allocate space for new tokens
        newContext[:, :, :, :, :npast] = Context  # Fill in with past context
    
    # Token embedding lookup (retrieve embeddings for the input tokens)
    inpATTN = Tensors['token_embd.weight'][tkn, :]
    
    # Loop through each layer of the model (each block)
    for nl in range(nLayer):
        # Retrieve weights for attention and feedforward networks (FFN) in this layer
        Attn_Norm = Tensors['blk.' + str(nl) + '.attn_norm.weight']  # Attention normalization weights
        FFN_Norm  = Tensors['blk.' + str(nl) + '.ffn_norm.weight']  # FFN normalization weights
        W_K = Tensors['blk.' + str(nl) + '.attn_k.weight']  # Attention key weights
        W_Q = Tensors['blk.' + str(nl) + '.attn_q.weight']  # Attention query weights
        W_V = Tensors['blk.' + str(nl) + '.attn_v.weight']  # Attention value weights
        Wo = Tensors['blk.' + str(nl) + '.attn_output.weight']  # Attention output weights
        WG = Tensors['blk.' + str(nl) + '.ffn_gate.weight']  # FFN gate weights
        Wup = Tensors['blk.' + str(nl) + '.ffn_up.weight']  # FFN upward layer weights
        Wdown = Tensors['blk.' + str(nl) + '.ffn_down.weight']  # FFN downward layer weights
        
        # Normalize input embeddings for this layer using RMS normalization
        inpA_norm = rms_norm(inpATTN, rms_Epsilon, Attn_Norm)
        
        # Calculate keys (K), queries (Q), and values (V) for the attention mechanism
        cur = inpA_norm.transpose()
        K = W_K @ cur
        Q = W_Q @ cur
        V = W_V @ cur
     
        # Reshape the K, Q, and V matrices to account for multiple heads
        K = K.reshape(nHead, nDIM, Ntkn)
        Q = Q.reshape(nHead, nDIM, Ntkn)
        V = V.reshape(nHead, nDIM, Ntkn)
        
        # Apply Rotary Positional Encoding (ROPE) on keys and queries
        [Mc, Ms] = matROPE(Ntkn, npast, nDIM, thscale)
        Q = ROPE(Q, Mc, Ms)
        K = ROPE(K, Mc, Ms)

        # Update the context with the new keys and values
        newContext[INDK, nl, :, :, npast:] = K
        newContext[INDV, nl, :, :, npast:] = V
        K = newContext[INDK, nl, :, :, :]  # Retrieve the updated keys
        V = newContext[INDV, nl, :, :, :]  # Retrieve the updated values
        
        # Compute attention using Q, K, and V
        A = Attention(Q, K, V, npast)
        A = A.reshape(nEMBD, Ntkn)
        A = Wo @ A  # Apply attention output weights
        A = A.transpose()
                
        # Feedforward network (FFN) processing with residual connection
        inpFFN = A + inpATTN  # Add residual connection
        inpFFN_Norm = rms_norm(inpFFN, rms_Epsilon, FFN_Norm)  # Normalize input for FFN

        cur = inpFFN_Norm.transpose()  # Prepare input for feedforward network

        G = WG @ cur  # Calculate gate output
        W = Wup @ cur  # Calculate upward FFN output
        tmp = W * SILU(G)  # Apply SILU activation function
        out = Wdown @ tmp  # Calculate downward FFN output

        inpATTN = out.transpose() + inpFFN  # Add FFN output to residual for the next layer

    # Output layer processing (after passing through layers)
    Wout = Tensors['output.weight']  # Output weights
    Out_Norm = Tensors['output_norm.weight']  # Output normalization weights

    # Normalize and calculate the final output
    out_norm = rms_norm(inpATTN[-1, :], rms_Epsilon, Out_Norm).transpose()
    out = Wout @ out_norm  # Compute final prediction
    out = out.reshape(-1)

    # Select top 10 predictions and apply softmax for probabilities
    I = np.argsort(out)[-10:][::-1]
    Pr = softmax(out[I])
    E = np.cumsum(Pr)
    Iout = sum(E < np.random.rand())  # Select a token stochastically based on cumulative proba
    TKout = I[Iout]  # Final output token

    return TKout, newContext  
