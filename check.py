#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:10:51 2024

@author: ctualle
"""

import numpy as np
import matplotlib.pyplot as plt

def check(tpy,name):
    
    file = open(name, 'r')
    f = file.read()
    file.close()
    
    f = f.replace(',','.')
    f = f.splitlines()
    f = [np.float32(i) for i in f ]
    
    t = tpy.shape
    N = t[0]*t[1]

    print(len(f),'=?',N)
    if len(f) != N:
        raise ValueError("Not the same size")
    
    a = np.resize(tpy,N)
    I = np.array(range(N))
    plt.plot(I,f,'b',I+100,a,'r')
    
    return np.linalg.norm(a-f)
