# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:42:25 2023

@author: Hasan Emre
"""

"""
    Sprite = ekran uzerinde hareket eden her seye sprite diyoruz. hepsini sprite_group tanimlayarak icinde olusturacagiz
"""
# pygame template
import pygame
import random

import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pickle





# window size
WIDTH = 360   # 360 a360 lik bir pencere olusturuyoruz
HEIGHT = 360
FPS = 1000000000000 # how fast game is

# colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)  # RGB
GREEN = (0,255,0)
BLUE = (0,0,255)

#  ekran uzerinde hareket eden her seye sprite diyoruz. hepsini sprite_group tanimlayarak icinde olusturacagiz
class Player(pygame.sprite.Sprite): # burada pygameden bir ust sinif miraz alarak bir subclass olusturuyoruz
    # sprite for thw player
    def __init__(self):
        super().__init__()   # super() kullanir gibi sut sinifin parametrelerini cekiyoruz
        self.image = pygame.Surface((20,20))   # 20 ye 20 bir yuzey olusturuyoruz ekran uzerinde
        self.image.fill(BLUE)   # ve bunun icini maviye boyuyoruz
        self.rect = self.image.get_rect()   # get_rect() metotu ile bu karenin etrafının gorunmez bir cizgi cekiyoruz
        self.radius = 10
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.rect.centerx = WIDTH/2   
        self.rect.bottom = HEIGHT -1
        self.speedx = 0
        
    def update(self, action):
        self.speedx = 0
        keystate = pygame.key.get_pressed()
        
        if keystate[pygame.K_LEFT] or action == 0:
            self.speedx = -7
        elif keystate[pygame.K_RIGHT] or action == 1:
            self.speedx = 7
        else:
            self.speedx = 0

        self.rect.x +=self.speedx     
        
        
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
            
        if self.rect.left < 0:
            self.rect.left = 0
            
    def getCoordinnates(self):
        return(self.rect.x, self.rect.y)
    
    
class Enemy(pygame.sprite.Sprite):  # dusman classi 
    def __init__(self): 
        super().__init__()
        self.image = pygame.Surface((10,10))  # 10 a 10 seklinde bir sekil olusturduk ve kirmizi rengini verdik
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.radius = 10
        pygame.draw.circle(self.image, WHITE, self.rect.center, self.radius)
        
        self.rect.x = random.randrange(0,WIDTH- self.rect.width)  # x ekseni boyunca yukaridan dusman cikabilir
        self.rect.y = random.randrange(2,6)  # y ekseni icinde 2ve 6 araliginda dusman olusturuluyor.
        
        self.speedx = 0
        self.speedy = 20 # x ekseni boyunca hareket edemeyecegi icin x e dogru 0 y ye dogru 10 luk bir hiz verdik

    def update(self):
        self.rect.x += self.speedx  # update kisminda güncellemeler yapılıyor speedler her seferinde guncelleniyor
        self.rect.y += self.speedy
            
        if self.rect.top > HEIGHT +10:    # dusmanlar yere deydiginde yok oluyor ve  yukaridan tekrar doguyor
            self.rect.x = random.randrange(0,WIDTH- self.rect.width)
            self.rect.y = random.randrange(2,6)
            self.speedy = 20
            
    def getCoordinnates(self):
          return(self.rect.x, self.rect.y)
            
      
class DQLAgent:
    
    def __init__(self):
        # parameter / hyperparameter
        self.state_size = 12 # distance [(playerx-m1x),(playery-m1y),(playerx-m2x),(playery-m2y),(playerx-m3x),(playery-m3y),(playerx-m4x),(playery-m4y),(playerx-m5x),(playery-m5y),(playerx-m6x),(playery-m6y)]
        self.action_size = 3  # right, left, no move
        
        
        self.gamma = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 1 # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen= 1000)
        
        self.model = self.build_model()
        self.target_model = self.build_model()
        
        
    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(64, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(64, activation = "relu"))
        model.add(Dense(32, activation = "relu"))
        
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
    
    
    def remember(self, state, action, reward, next_state, done):
        #storage
        self.memory.append((state, action, reward, next_state, done))
    
    
    def act(self,state):
        # acting: explore and exploit
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # env.action_space.sample() digerine göre ufak bir degisiklik var
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])
        
        if len(not_done_indices[0]) > 0:
            predict_sprime = self.model.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.target_model.predict(np.vstack(minibatch[:, 3]))
        
            y[not_done_indices] += np.multiply(self.gamma, np.squeeze(predict_sprime_target[not_done_indices, np.argmax(predict_sprime[not_done_indices], axis=1)]))
        
        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.model.predict(np.vstack(minibatch[:, 0]))
        y_target[range(batch_size), actions] = y
        
        self.model.fit(np.vstack(minibatch[:, 0]), y_target, epochs=1, verbose=0)

            
    
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
            
class Env(pygame.sprite.Sprite):   # bir tane envirenmont classi olusturuyoruz.
    def __init__(self):
        super().__init__()
        self.all_sprite = pygame.sprite.Group()  # disarida tanimladigimiz objeleri buraya aliyoruz
        self.enemy = pygame.sprite.Group()
        
        self.player = Player()
        self.all_sprite.add(self.player) 
        
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.m3 = Enemy()
        self.m4 = Enemy()
        self.m5 = Enemy()
        self.m6 = Enemy()
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.all_sprite.add(self.m3)
        self.all_sprite.add(self.m4)
        self.all_sprite.add(self.m5)
        self.all_sprite.add(self.m6)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.enemy.add(self.m3)
        self.enemy.add(self.m4)
        self.enemy.add(self.m5)
        self.enemy.add(self.m6)
        
        self.reward = 0 
        self.total_reward = 0
        self.done = False
        self.agent = DQLAgent()        

    
    def findDistance(self,a,b):  # enemy ile player arasındaki mesafeyi hesaplar
        return a-b
    
    def step(self,action):
        state_list = []
        
        # update
        self.player.update(action)
        self.enemy.update()
        
        #☺ get coordinate             koordinatlarini aldik 
        next_player_state = self.player.getCoordinnates()
        next_m1_state = self.m1.getCoordinnates()
        next_m2_state = self.m2.getCoordinnates()
        next_m3_state = self.m3.getCoordinnates()
        next_m4_state = self.m4.getCoordinnates()
        next_m5_state = self.m5.getCoordinnates()
        next_m6_state = self.m6.getCoordinnates()

        
        # find distance
        state_list.append(self.findDistance(next_player_state[0], next_m1_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m1_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m2_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m2_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m3_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m3_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m4_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m4_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m5_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m5_state[1]))
        state_list.append(self.findDistance(next_player_state[0], next_m6_state[0]))
        state_list.append(self.findDistance(next_player_state[1], next_m6_state[1]))
        
        return [state_list]
        

    def initialStates(self):
        self.all_sprite = pygame.sprite.Group()  # agent haric her seyi bastan yaratiyoruz
        self.enemy = pygame.sprite.Group()
        
        self.player = Player()
        self.all_sprite.add(self.player) 
        
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.m3 = Enemy()
        self.m4 = Enemy()
        self.m5 = Enemy()
        self.m6 = Enemy()
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.all_sprite.add(self.m3)
        self.all_sprite.add(self.m4)
        self.all_sprite.add(self.m5)
        self.all_sprite.add(self.m6)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.enemy.add(self.m3)
        self.enemy.add(self.m4)
        self.enemy.add(self.m5)
        self.enemy.add(self.m6)
        
        self.reward = 0 
        self.total_reward = 0
        self.done = False
        
        
        state_list =[]
        #☺ get coordinate             koordinatlarini aldik 
        player_state = self.player.getCoordinnates()
        m1_state = self.m1.getCoordinnates()
        m2_state = self.m2.getCoordinnates()
        m3_state = self.m3.getCoordinnates()
        m4_state = self.m4.getCoordinnates()
        m5_state = self.m5.getCoordinnates()
        m6_state = self.m6.getCoordinnates()
        
        # find distance
        state_list.append(self.findDistance(player_state[0], m1_state[0]))
        state_list.append(self.findDistance(player_state[1], m1_state[1]))
        state_list.append(self.findDistance(player_state[0], m2_state[0]))
        state_list.append(self.findDistance(player_state[1], m2_state[1]))
        state_list.append(self.findDistance(player_state[0], m3_state[0]))
        state_list.append(self.findDistance(player_state[1], m3_state[1]))
        state_list.append(self.findDistance(player_state[0], m4_state[0]))
        state_list.append(self.findDistance(player_state[1], m4_state[1]))
        state_list.append(self.findDistance(player_state[0], m5_state[0]))
        state_list.append(self.findDistance(player_state[1], m5_state[1]))
        state_list.append(self.findDistance(player_state[0], m6_state[0]))
        state_list.append(self.findDistance(player_state[1], m6_state[1]))
        
        return [state_list]
        
    
    def run(self):
        # game loop
        state = self.initialStates()
        running = True
        batch_size = 24
        
        # kaydetme kısmı ama yapamadim 
        # def save_model(agent):
        #     with open('agent_model.pkl', 'wb') as f:
        #         pickle.dump(agent, f)
        
        # def load_model():
        #     with open('agent_model.pkl', 'rb') as f:
        #         agent = pickle.load(f)
        #     return agent
        
        # # ...
        
        # # # Modeli kaydet
        # # save_model(self.agent)
        
        # # # ...
        
        # # # Modeli yükle
        # # self.agent = load_model()
        while running:
            self.reward = 2
            # keep loop running at the right speed (dongunun dogru hizda calismasini saglayin)
            clock.tick(FPS) # oyunun hizi = 30
            
                        
            # process input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:   # burada QUIT ile oyunun dongusunden cikiyor 
                    running = False
                    
            # update
            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward
            
                    
            hits = pygame.sprite.spritecollide(self.player,self.enemy,False,pygame.sprite.collide_circle)
            if hits: 
                self.reward = -150
                self.total_reward += self.reward
                self.done = True
                running = False
                print("Total Reward: ",self.total_reward)
                
            
            
            #storage    
            self.agent.remember(state, action, self.reward, next_state, self.done)
                
        
            # udpate state
            state = next_state
            
            # training
            self.agent.replay(batch_size)
            
            # epsilon greedy
            self.agent.adaptiveEGreedy()
            
            
             # draw / render(show)
            screen.fill(GREEN)
            self.all_sprite.draw(screen)
             
             # after drawing flip display
            pygame.display.flip()     # ekran duzenleme kisminda yaptigimiz değisiklikleri burada ceviriyoruz
             
            
          
        pygame.quit()  # burada tamamen oyunu kapatiyoruz 



if __name__ == "__main__":
    env = Env()
    liste = []
    t = 0
    while True:
        t += 1
        print("Episode: ",t)
        liste.append(env.total_reward)
        8
        # initialize pygame and create window
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT))  # yukarida olusturdugumuz boyutları burada burada isleme tutuyoruz
        pygame.display.set_caption("RL Game")  # pencereye bir baslik yazıyoruz
        clock = pygame.time.Clock()

        env.run()


# sprite 
# all_sprite = pygame.sprite.Group()  # burada objeler olusturuyoruz
# enemy = pygame.sprite.Group()
# player = Player()
# m1 = Enemy()
# m2 = Enemy()
# m3 = Enemy()
# m4 = Enemy()
# m5 = Enemy()
# m6 = Enemy()

# all_sprite.add(player)  # player yarattik
# all_sprite.add(m1)
# all_sprite.add(m2)
# all_sprite.add(m3)
# all_sprite.add(m4)
# all_sprite.add(m5)
# all_sprite.add(m6)
# enemy.add(m1)
# enemy.add(m2)
# enemy.add(m3)
# enemy.add(m4)
# enemy.add(m5)
# enemy.add(m6)

# game loop

# running = True
# while running:
#     # keep loop running at the right speed (dongunun dogru hizda calismasini saglayin)
#     clock.tick(FPS) # oyunun hizi = 30
    
#     # process input
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:   # burada QUIT ile oyunun dongusunden cikiyor 
#             running = False
            
        
    # update
    # all_sprite.update()
    
    # hits = pygame.sprite.spritecollide(player,enemy,False,pygame.sprite.collide_circle)
    # if hits: 
    #     running = False
    #     print("Game over")
        
    
    # # draw / render(show)
    # screen.fill(GREEN)
    # all_sprite.draw(screen)
    
    # # after drawing flip display
    # pygame.display.flip()     # ekran duzenleme kisminda yaptigimiz değisiklikleri burada ceviriyoruz
    
    
  
# pygame.quit()  # burada tamamen oyunu kapatiyoruz 


























