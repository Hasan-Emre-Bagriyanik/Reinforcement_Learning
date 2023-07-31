# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:26:05 2023

@author: Hasan Emre
"""

import pygame
import random


# burada sabit degiskenleri ayarlıyoruz 
FPS = 60
# pencere boyutu
WINDOW_WİDTH = 400
WINDOW_HEIGTH = 420
GAME_HEIGTH = 400
# paddle boyutu 
PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_BUFFER = 15
# top boyutu
BALL_WIDTH = 20
BALL_HEIGHT = 20

# paddle ve top hizilari
PADDLE_SPEED = 3
BALL_X_SPEED = 1
BALL_Y_SPEED = 1

# renkler
WHITE = (255,255,255)
BLACK = (0,0,0)

# pencere olusturma 
screen = pygame.display.set_mode((WINDOW_WİDTH,WINDOW_HEIGTH))


# paddle imizi olusturuyoruz
def drawPaddle(switch,paddleYPos):

    if switch == "left":  # paddle sola gidercekse konumu
        paddle = pygame.Rect(PADDLE_BUFFER, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
        
    elif switch == "right": # paddle saga gidercekse konumu
        paddle = pygame.Rect(WINDOW_WİDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    
    # ve paddle olusturuyoruz
    pygame.draw.rect(screen, WHITE, paddle)
    
    
# topumuzu ayarlıyoruz
def drawBall(ballXPos,ballYPos):
    # topumuzun konumunu belirliyoruz
    ball = pygame.Rect(ballXPos, ballYPos, BALL_WIDTH, BALL_HEIGHT)
    # topumuzu olusturyoruz
    pygame.draw.rect(screen, WHITE, ball)
    
    
    
    
    
def updatePaddle(switch, action, paddleYPos, ballYPos):
    dft = 7.5
    
    # agent
    if switch == "left":
        if action == 1:
            paddleYPos = paddleYPos - PADDLE_SPEED*dft
            
        if action == 2:
            paddleYPos = paddleYPos + PADDLE_SPEED*dft
            
            
        if paddleYPos < 0:
            paddleYPos = 0
        
        if paddleYPos > GAME_HEIGTH - PADDLE_HEIGHT:
            paddleYPos = GAME_HEIGTH - PADDLE_HEIGHT
            
            
    elif switch == "right":
        if paddleYPos + PADDLE_HEIGHT/2  < ballYPos + BALL_HEIGHT/2:
            paddleYPos = paddleYPos + PADDLE_SPEED*dft
            
        if paddleYPos + PADDLE_HEIGHT/2  > ballYPos + BALL_HEIGHT/2:
            paddleYPos = paddleYPos - PADDLE_SPEED*dft
            
            
        if paddleYPos < 0:
            paddleYPos = 0
        
        if paddleYPos > GAME_HEIGTH - PADDLE_HEIGHT:
            paddleYPos = GAME_HEIGTH - PADDLE_HEIGHT
    
    return paddleYPos
    




def updateBall(paddle1YPos,paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection,DeltaFrameTime):
    dft = 7.5
    
    ballXPos = ballXPos + ballXDirection*BALL_X_SPEED*dft
    ballYPos = ballYPos + ballYDirection*BALL_Y_SPEED*dft
    
    score = -0.05
    
    # agent paddle1imiz ile carpismalar
    if ((ballXPos <= (PADDLE_BUFFER + PADDLE_WIDTH)) and ((ballYPos + BALL_HEIGHT) >= paddle1YPos) and (ballYPos <= (paddle1YPos + PADDLE_HEIGHT))) and (ballXDirection == -1):
        ballXDirection = 1
        score = 10
        
    elif (ballXPos <= 0):
        
        ballXDirection = paddle1YPos
        score = -10
        
        return [score,ballXPos,ballYPos,ballXDirection,ballYDirection]
        
    # agent olarak kullanmayan paddle2imiz icin top ile carpismalari
    if ((ballXPos >= (WINDOW_WİDTH - PADDLE_WIDTH)) and ((ballYPos + BALL_HEIGHT) >=paddle2YPos) and (ballYPos <= (paddle2YPos + PADDLE_HEIGHT)) and (ballXDirection ==1)):
        ballXDirection = -1
        
    elif (ballXPos >= WINDOW_WİDTH - BALL_WIDTH):
        ballXDirection = -1
        return [score, ballXPos,ballYPos, ballXDirection, ballYDirection]
    
    
    
    # topumuz yukarı veya asagiya carparsa diye geri donmesi icin
    if ballYPos <=0:
        ballYPos = 0
        ballYDirection = 1
        
    elif ballYPos >=GAME_HEIGTH - BALL_HEIGHT:
        ballYPos = GAME_HEIGTH - BALL_HEIGHT
        ballYDirection = -1
        
        
    return [score, ballXPos,ballYPos, ballXDirection, ballYDirection]

    




class PongGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Pong DCQL Env")  # pencereye baslik ekledik
        # bunlar baslangictaki ilk konumları olacak
        self.paddle1YPos = GAME_HEIGTH/2 - PADDLE_HEIGHT/2   # iki paddle nin da konumlarını belirliyoruz
        self.paddle2YPos = GAME_HEIGTH/2 - PADDLE_HEIGHT/2
        
        self.ballXPos = WINDOW_WİDTH/2 -10   # burada topu tam orta noktada olacak sekilde ayarladik
        
        self.clock = pygame.time.Clock()
        
        self.GScore = 0.0
        
        self.ballXDirection = random.sample([-1,1], 1)[0]
        self.ballYDirection = random.sample([-1,1], 1)[0]
        
        self.ballYPos = random.randint(0, 9)*(WINDOW_HEIGTH - BALL_HEIGHT)/9 # topumuzu orta noktada rastgele konumlandiriyoruz
        
        
        
        
    def InitialDisplay(self):
        
        pygame.event.pump()  
        
        screen.fill(BLACK)  # pencereyi boyadik
        
        drawPaddle("left",self.paddle1YPos) # fonksiyonlarımızı yukaridan cagirdik
        drawPaddle("right",self.paddle2YPos)
        
        drawBall(self.ballXPos,self.ballYPos)
        
        pygame.display.flip()
        
        
        
    
    def PlayNextMove(self,action):
        
        DeltaFrameTime = self.clock.tick(FPS)
        
        pygame.event.pump()
        
        score = 0
        
        screen.fill(BLACK)
        
        self.paddle1YPos =  updatePaddle("left", action, self.paddle1YPos,self.ballYPos)
        drawPaddle("left", self.paddle1YPos)

        self.paddle2YPos =  updatePaddle("right", action, self.paddle2YPos,self.ballYPos)
        drawPaddle("right", self.paddle2YPos)

        
        [score, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection] = updateBall(self.paddle1YPos,self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection,DeltaFrameTime)


        drawBall(self.ballXPos, self.ballYPos)
        
        if(score <0.5 or score <-0.5):
            self.GScore = self.GScore*0.9 + 0.1*score
            
            
        ScreenImage = pygame.surfarray.array3d(pygame.display.get_surface())
        
        pygame.display.flip()
        
        return [score, ScreenImage]
        















