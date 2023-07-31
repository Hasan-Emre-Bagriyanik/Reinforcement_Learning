# -*- coding: utf-8 -*-
"""
Created on Wed May 24 18:59:51 2023

@author: Hasan Emre
"""

import DCQL_Agent
import DCQL_Pong

import numpy as np
import skimage as skimage
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

TOTAL_TrainTime = 100000

IMG_HEIGHT = 40
IMG_WIDTH = 40
IMG_HISTORY = 4

def ProcessGameImage(RawImage):
    
    GreyImage = skimage.color.rgb2gray(RawImage)
    
    CroppedImage = GreyImage[0:400,0:400]
    
    ReducedImage = skimage.transform.resize(CroppedImage,(IMG_HEIGHT,IMG_WIDTH))
    ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range= (0,255))
    
    ReducedImage = ReducedImage / 128
    
    return ReducedImage
    

def TrainExperiment():
    
    Train_history = []
    
    TheGame = DCQL_Pong.PongGame()

    TheGame.InitialDisplay()
    
    
    TheAgent = DCQL_Agent.Agent()

    BestAction = 0
    
    [initialScore, initialScreenImage] = TheGame.PlayNextMove(BestAction)
    initailGameImage = ProcessGameImage(initialScreenImage)

    GameState = np.stack((initailGameImage,initailGameImage,initailGameImage,initailGameImage),axis = 2)
    
    GameState = GameState.reshape(1, GameState.shape[0],GameState.shape[1], GameState.shape[2])

    for i in range(TOTAL_TrainTime):
        BestAction = TheAgent.FindBestAct(GameState)
        [ReturnScore, NewScreenImage] = TheGame.PlayNextMove(BestAction)

        NewScreenImage = ProcessGameImage(NewScreenImage)
        
        NewScreenImage = NewScreenImage.reshape(1,NewScreenImage.shape[0], NewScreenImage.shape[1],1)
        
        nextState = np.append(NewScreenImage, GameState[:,:,:,:3], axis=3)
        
        TheAgent.CaptureSample((GameState,BestAction,ReturnScore,nextState))
        
        TheAgent.Process()

        GameState = nextState
        
        if i %250 ==0:
            print("Train time: {},  Game score: {}".format(i, TheGame.GScore))
            Train_history.append(TheGame.GScore)



TrainExperiment()





