#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Pattern functions

@author: Filippo Squillace
@date: 04/12/2011
@version: 0.3.0
"""
import numpy as np

def lightWeightSpaceship():
    """
    Puts the lightweight spaceship right in the grid starting from icoords
    """
    m = np.array([False for i in range(4*5)]).reshape(4, 5)
    m[0,0] = True
    m[0,3] = True
    m[2,0] = True
    m[3,1] = True
    m[3,2] = True
    m[3,3] = True
    m[3,4] = True
    m[2,4] = True
    m[1,4] = True

    return m

def middleWeightSpaceship():
    """
    Puts the middleweight spaceship right in the grid starting from icoords
    """
    m = np.array([False for i in range(5*6)]).reshape(5, 6)
    m[1,0] = True
    m[3,0] = True
    m[4,1] = True
    m[4,2] = True
    m[4,3] = True
    m[4,4] = True
    m[4,5] = True
    m[3,5] = True
    m[2,5] = True
    m[1,4] = True
    m[0,2] = True

    return m

def heavyWeightSpaceship():
    """
    Puts the heavyweight spaceship
    """
    m = np.array([False for i in range(5*7)]).reshape(5,7)
    m[1,0] = True
    m[3,0] = True
    m[4,1] = True
    m[4,2] = True
    m[4,3] = True
    m[4,4] = True
    m[4,5] = True
    m[4,6] = True
    m[3,6] = True
    m[2,6] = True
    m[1,5] = True
    m[0,2] = True
    m[0,3] = True

    return m

def glider():
    """
    Puts the glider
    """
    m = np.array([False for i in range(3*3)]).reshape(3,3)
    m[2,0] = True
    m[2,1] = True
    m[2,2] = True
    m[1,2] = True
    m[0,1] = True

    return m


def eater():
    """
    Puts the eater
    """
    m = np.array([False for i in range(4*4)]).reshape(4,4)
    m[3,0] = True
    m[3,1] = True
    m[2,1] = True
    m[1,1] = True
    m[0,2] = True
    m[0,3] = True
    m[1,3] = True

    return m


def pulsar():
    """
    Puts the pulsar
    """
    m = np.array([False for i in range(7*3)]).reshape(7,3)
    m[0,1] = True
    m[1,0] = True
    m[1,2] = True
    m[2,1] = True
    m[3,0] = True
    m[3,2] = True
    m[4,1] = True
    m[5,0] = True
    m[5,2] = True
    m[6,1] = True

    return m


def fPentomino():
    """
    Puts the fPentomino
    """
    m = np.array([False for i in range(3*3)]).reshape(3,3)
    m[1,0] = True
    m[0,1] = True
    m[1,1] = True
    m[2,1] = True
    m[0,2] = True

    return m


def aCorn():
    """
    Puts the aCorn
    """
    m = np.array([False for i in range(3*7)]).reshape(3,7)
    m[2,0] = True
    m[2,1] = True
    m[0,1] = True
    m[1,3] = True
    m[2,4] = True
    m[2,5] = True
    m[2,6] = True

    return m

def gosperGliderGun():
    """
    Puts the 
    """
    m = np.array([False for i in range(9*36)]).reshape(9,36)
    m[4,0] = True
    m[4,1] = True
    m[5,0] = True
    m[5,1] = True
    m[4,10] = True
    m[5,10] = True
    m[6,10] = True
    m[7,11] = True
    m[8,12] = True
    m[8,13] = True
    m[3,11] = True
    m[2,12] = True
    m[2,13] = True
    m[5,14] = True
    m[3,15] = True
    m[4,16] = True
    m[5,16] = True
    m[6,16] = True
    m[7,15] = True
    m[5,17] = True
    m[4,20] = True
    m[3,20] = True
    m[2,20] = True
    m[2,21] = True
    m[3,21] = True
    m[4,21] = True
    m[5,22] = True
    m[1,22] = True
    m[1,24] = True
    m[0,24] = True
    m[5,24] = True
    m[6,24] = True
    m[3,34] = True
    m[2,34] = True
    m[2,35] = True
    m[3,35] = True

    return m


def blockLaying():
    """
    Puts the 
    """
    m = np.array([False for i in range(6*8)]).reshape(6,8)
    m[5,0] = True
    m[5,2] = True
    m[4,2] = True
    m[3,4] = True
    m[2,4] = True
    m[1,4] = True
    m[0,6] = True
    m[1,6] = True
    m[2,6] = True
    m[1,7] = True

    return m

def blockLaying2():
    """
    Puts the 
    """
    m = np.array([False for i in range(5*5)]).reshape(5,5)
    m[0,0] = True
    m[0,1] = True
    m[0,2] = True
    m[0,4] = True
    m[1,0] = True
    m[2,3] = True
    m[2,4] = True
    m[3,1] = True
    m[3,2] = True
    m[3,4] = True
    m[4,0] = True
    m[4,2] = True
    m[4,4] = True

    return m

def thin():
    """
    Puts the 
    """
    m = np.array([False for i in range(1*39)]).reshape(1,39)
    m[0,0] = True
    m[0,1] = True
    m[0,2] = True
    m[0,3] = True
    m[0,4] = True
    m[0,5] = True
    m[0,6] = True
    m[0,7] = True
    m[0,9] = True
    m[0,10] = True
    m[0,11] = True
    m[0,12] = True
    m[0,13] = True
    m[0,17] = True
    m[0,18] = True
    m[0,19] = True
    m[0,26] = True
    m[0,27] = True
    m[0,28] = True
    m[0,29] = True
    m[0,30] = True
    m[0,31] = True
    m[0,32] = True
    m[0,34] = True
    m[0,35] = True
    m[0,36] = True
    m[0,37] = True
    m[0,38] = True

    return m

def frog():
    """
    Puts the 
    """
    m = np.array([False for i in range(6*16)]).reshape(6,16)
    m[3,0] = True
    m[4,0] = True
    m[3,1] = True
    m[3,2] = True
    m[4,3] = True
    m[5,1] = True
    m[5,2] = True
    m[0,2] = True
    m[0,3] = True
    m[1,2] = True
    m[0,12] = True
    m[0,13] = True
    m[1,13] = True
    m[3,13] = True
    m[3,14] = True
    m[3,15] = True
    m[4,12] = True
    m[4,15] = True
    m[5,13] = True
    m[5,14] = True
    m[2,3] = True
    m[2,4] = True
    m[2,5] = True
    m[2,6] = True
    m[2,7] = True
    m[2,8] = True
    m[2,9] = True
    m[2,10] = True
    m[2,11] = True
    m[2,12] = True
    m[1,5] = True
    m[1,6] = True
    m[1,7] = True
    m[1,8] = True
    m[1,9] = True
    m[1,10] = True
    m[0,6] = True
    m[0,7] = True
    m[0,8] = True
    m[0,9] = True

    return m

def spark():
    """
    """
    m = np.array([False for i in range(7*8)]).reshape(7,8)
    m[5,0] = True
    m[6,0] = True
    m[6,1] = True
    m[6,7] = True
    m[6,6] = True
    m[5,7] = True
    m[5,2] = True
    m[4,2] = True
    m[5,5] = True
    m[4,5] = True
    m[3,1] = True
    m[2,1] = True
    m[1,1] = True
    m[3,6] = True
    m[2,6] = True
    m[1,6] = True
    m[0,3] = True
    m[0,4] = True

    return m

def puffers():
    """
    """
    m = np.array([False for i in range(18*5)]).reshape(18,5)
    m[2,0] = True
    m[3,1] = True
    m[3,2] = True
    m[3,3] = True
    m[3,4] = True
    m[2,4] = True
    m[1,4] = True
    m[0,3] = True
    m[7,0] = True
    m[8,1] = True
    m[8,2] = True
    m[9,2] = True
    m[10,2] = True
    m[11,1] = True
    m[16,0] = True
    m[17,1] = True
    m[17,2] = True
    m[17,3] = True
    m[17,4] = True
    m[16,4] = True
    m[15,4] = True
    m[14,3] = True

    return m

