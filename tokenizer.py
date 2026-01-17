#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:26:16 2024

@author: ctualle
"""

SPECIAL_CHARACTER = '▁'


def addBigram(jprev,jnext,Symbols,Work,Model):
    txt1 = Symbols[jprev][0]
    txt2 = Symbols[jnext][0]
    
    if (txt1=='') or (txt2==''):
        return
    txt = txt1+txt2
    #print('try: '+txt)
    try:
        ind = Model.Key["tokenizer.ggml.tokens"].index(txt)
    except:
        return
    #print('Found: append ',ind)
    Work.append([Model.Key["tokenizer.ggml.scores"][ind],jprev,jnext,txt])
    
    return


def tokenize(texte,Model):
    L = texte.split()
    Txt = ''
    for k in L:
        Txt += SPECIAL_CHARACTER + k
    
    Symbols = []
    TxtLength = len(Txt)
    for i in range(TxtLength):
        Symbols.append([Txt[i],i-1,i+1])
     
        
    Work = []
    for j in range(len(Txt)-1):
        addBigram(j,j+1,Symbols,Work,Model)
    
    while (len(Work)>0):
        max_value = max(Work, key=lambda item: item[0])
        max_ind = Work.index(max_value)
        Work.pop(max_ind)   # enlever le score maximal de la liste pop
        
        jmin = max_value[1]
        jmax = max_value[2]
        txt = max_value[3]
        txt1 = Symbols[jmin][0]  # merging of the symbols
        txt2 = Symbols[jmax][0]
        if (txt==txt1+txt2):
            #print('Extraction de ',max_value)
            Symbols[jmin][0] = txt1+txt2
            Symbols[jmax][0] = ''  
            new_jmax = Symbols[jmax][2]
            Symbols[jmin][2] = new_jmax
            
            njmin = Symbols[jmin][1]
            if (njmin>=0):
                addBigram(njmin,jmin,Symbols,Work,Model)
            if (new_jmax<TxtLength):
                Symbols[new_jmax][1] = jmin
                addBigram(jmin,new_jmax,Symbols,Work,Model)
            #print(Symbols)
            
    Token = []
    for k in Symbols:
        txt = k[0] 
        if txt != '':
            Token.append(Model.Key["tokenizer.ggml.tokens"].index(txt))
    return Token



           
    
