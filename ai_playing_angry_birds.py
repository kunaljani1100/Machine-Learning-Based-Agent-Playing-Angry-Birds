# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:37:05 2020

@author: Kunal Jani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mt

y_location_of_pigs=[0,0,0,0,0]
x_location_of_pigs=[]
destroyed=[False,False,False,False,False]
num_pigs=5
for i in range(0,num_pigs):
    x_location_of_pigs.append(np.random.uniform(15,40))
plt.scatter(x_location_of_pigs,y_location_of_pigs,color='blue')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.xlim(0,30)
plt.ylim(0,10)
plt.show()
all_destroyed=True
plays=0
while(plays<9):
    all_destroyed=True
    for i in range(len(destroyed)):
        if(destroyed[i]==False):
            all_destroyed=False
            break
    if(all_destroyed):
        break
    datasets=[]
    for i in range(1,6):
        datasets.append(pd.read_csv('game_results'+str(i)+'.csv'))
    x_train=datasets[len(x_location_of_pigs)-1].iloc[:,:-1].values
    y_train=datasets[len(x_location_of_pigs)-1].iloc[:,-1].values
    
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 1, metric = 'minkowski', p = 2)
    classifier.fit(x_train,y_train)
    retest=True
    for trials in range(15):
        #Initializing the parameters for the projectile motion.
        u=np.random.uniform(10,30) #Initial velocity
        g=10 #Acceleration due to gravity
        angle=np.random.uniform(0,90) #Angle of projection
        theta=(angle*3.14)/180
        t=(2*u*np.sin(theta))/g #Total time of flight in case of a perfectly parabolic trajectory
        dt=0.01
        
        test_value=[]
        test_value.append(u)
        test_value.append(theta)
        for i in range(len(x_location_of_pigs)):
            test_value.append(x_location_of_pigs[i])
        
        test_value=np.array(test_value)
        test_values=[]
        test_values.append(test_value)     
        y_pred = classifier.predict(test_values)
        
        if(y_pred[0]>=1):
            retest=False
        
        if(retest==False):
            break
    
    time=[]
    for i in range(int(t/dt)):
      time.append(dt*i)
      
    x=[]
    y=[]
    x_init=0
    y_init=0
    
    for i in range(int(t/dt)):
        if(i==0):
          x.append(x_init)
          y.append(y_init)
        else:  
          x.append(x[i-1]+(u*np.cos(theta)*dt))
          y.append(y[i-1]+(u*np.sin(theta)-g*time[i])*dt)
    
    plt.plot(x,y)
    '''
    plt.arrow(0,0,6*np.cos(theta),6*np.sin(theta),length_includes_head=True,head_width=0.1, head_length=0.2)
    plt.arrow(0,0,6,0,length_includes_head=True,head_width=0.1, head_length=0.2)
    '''
    #plt.text(0.5,0.1,r'$\theta$')
    '''
    p1=[30,34.6]
    p2=[0,0]
    plt.plot(p1,p2)
    '''
    #plt.text(32,0.2,'d')
    print(x[len(x)-1])
    num_pigs_destroyed=0
    for i in range(0,len(x_location_of_pigs)):
        if((destroyed[i]==False) and abs(x[len(x)-1]-x_location_of_pigs[i])<1):
            destroyed[i]=True
            num_pigs_destroyed=num_pigs_destroyed+1
    
    for i in range(len(x_location_of_pigs)):
        if(destroyed[i]==True):
            plt.scatter(x_location_of_pigs[i],y_location_of_pigs[i],color='red',label='Destroyed')
        else:
            plt.scatter(x_location_of_pigs[i],y_location_of_pigs[i],color='blue',label='Not Destroyed')
    
    import csv
    with open('game_results'+str(len(x_location_of_pigs))+'.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if(len(x_location_of_pigs)==1):
            writer.writerow([u,angle,x_location_of_pigs[0],num_pigs_destroyed])
        if(len(x_location_of_pigs)==2):
            writer.writerow([u,angle,x_location_of_pigs[0],x_location_of_pigs[1],num_pigs_destroyed])
        if(len(x_location_of_pigs)==3):
            writer.writerow([u,angle,x_location_of_pigs[0],x_location_of_pigs[1],x_location_of_pigs[2],num_pigs_destroyed])
        if(len(x_location_of_pigs)==4):
            writer.writerow([u,angle,x_location_of_pigs[0],x_location_of_pigs[1],x_location_of_pigs[2],x_location_of_pigs[3],num_pigs_destroyed])
        if(len(x_location_of_pigs)==5):
            writer.writerow([u,angle,x_location_of_pigs[0],x_location_of_pigs[1],x_location_of_pigs[2],x_location_of_pigs[3],x_location_of_pigs[4],num_pigs_destroyed])
    
    i=0
    while(i<len(x_location_of_pigs)):
        if(destroyed[i]==True):
            x_location_of_pigs.pop(i)
            y_location_of_pigs.pop(i)
            destroyed.pop(i)
            i=i-1
        i=i+1
    
    plt.title('Trajectory of an angry bird')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.xlim(0,45)
    plt.ylim(0,10)
    plt.show()
    plays=plays+1