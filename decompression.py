#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:05:33 2024

@author: claire
"""

import struct
import numpy as np
import Loader

# Function to convert a 16-bit floating point (fp16) number into a 32-bit floating point (fp32) number
def fp16_to_fp32(fp16_as_int):
    """
    Converts a half-precision floating point number (fp16) stored as an integer into 
    a single-precision floating point number (fp32).

    Parameters:
        fp16_as_int(int): The fp16 number represented as a 16-bit integer.
    
    Returns:
        float : The corresponding 32-bit floating point number.
    """
    
    # Extract sign (1 bit), exponent (5 bits), and fraction (10 bits) from fp16
    sign = (fp16_as_int & 0x8000) >> 15
    exponent = (fp16_as_int & 0x7C00) >> 10
    fraction = fp16_as_int & 0x03FF

    # Compute the fp32 components based on the fp16 format
    if exponent == 0:  
        if fraction == 0:
            # Zero case (fp16 represents 0)
            fp32_exponent = 0
            fp32_fraction = 0
        else:
            # Subnormal case (small values)
            exponent_shift = 10 - fraction.bit_length()
            fp32_exponent = 127 - 15 - exponent_shift + 1
            fp32_fraction = fraction << (23 - 10 + exponent_shift)
    elif exponent == 0x1F:
        # Infinity or NaN case (exponent all ones)
        fp32_exponent = 255
        fp32_fraction = fraction << (23 - 10)
    else:
        # Normal case (typical numbers)
        fp32_exponent = exponent + (127 - 15)
        fp32_fraction = fraction << (23 - 10)

    # Reassemble the fp32 number by combining sign, exponent, and fraction
    fp32_as_int = (sign << 31) | (fp32_exponent << 23) | fp32_fraction
    
    # Pack the 32-bit integer as bytes and unpack it as a float
    packed_fp32 = struct.pack('I', fp32_as_int)
    fp32 = struct.unpack('f', packed_fp32)[0]

    return fp32

# Precompute all possible fp16 to fp32 conversions for every possible 16-bit value (65536 values)
FP16_TO_FP32 = [fp16_to_fp32(i) for i in range(65536)]

# Function to dequantize a Q8_0 quantized buffer
def dequantize_q8_0(buf, size, bloc_size, raw_size, typestruct):
    """
    Dequantizes a quantized Q8_0 tensor buffer into fp32 (floating-point) numbers.
    
    Parameters:
        buf : bytes
            The input buffer containing quantized data.
        size : int
            The number of floating-point values to output.
        bloc_size : int
            The block size used in quantization.
        raw_size : int
            The raw size of a block, including the scale and quantized data.
        typestruct : str
            The struct format used to unpack the data.
    
    Returns:
        np.array : The dequantized floating-point numbers.
    """
    
    y = np.empty(size, dtype='float32')  # Initialize an empty array to hold the dequantized values
    cnt_y = 0
    
    # Iterate through the buffer in chunks (each chunk corresponds to a block of quantized data)
    for cnt_buf in range(0, len(buf), raw_size):
        # Unpack the scale (d16) and quantized values (s) from the buffer
        [d16, s] = struct.unpack(typestruct, buf[cnt_buf:cnt_buf + raw_size])
        
        # Calculate the index range for the current block in the output array
        i_cnt_y = cnt_y + bloc_size
        
        # Dequantize the block and store the result in the output array
        y[cnt_y:i_cnt_y] = FP16_TO_FP32[d16] * np.array(list(s)).astype('int8')
        
        cnt_y = i_cnt_y  # Update the output index
        
    return y

# Function to extract tensor data from a block tensor
def extract(blkTensor):
    """
    Extracts and dequantizes tensor data from a given block tensor.
    
    Parameters:
        blkTensor : dict
            Dictionary containing the tensor type and data.
    
    Returns:
        np.array : The dequantized or unpacked tensor data.
    """
    
    Ttype = blkTensor['tensor_type']  # Type of the tensor ('Q8_0' or 'F32')
    Tsize = blkTensor['taille_dimensions'][::-1]  # Get tensor size and reverse the dimensions
    N = np.prod(Tsize)  # Compute the total number of elements in the tensor
    
    # Dequantize or unpack based on tensor type
    if Ttype == 'Q8_0':
        [bloc_size, raw_size, typestruct] = [32, 34, '<H32s']  # Parameters for Q8_0 tensor type
        data = dequantize_q8_0(blkTensor['data'], N, bloc_size, raw_size, typestruct)  
    elif Ttype == 'F32':
        # Directly unpack the floating-point data
        data = np.array(struct.unpack('<' + str(N) + 'f', blkTensor['data']), dtype='float32')
    
    data = data.reshape(Tsize)  
    
    return data  
