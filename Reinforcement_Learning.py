# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 09:09:39 2019

@author: jgraham
"""
#%%
import pandas as pd
import numpy as np
import dask.dataframe as dd 
import networkx as nx
import pylab as plt 
#%%
points_list = [(0,1),(1,5),(5,6),(5,4),(1,2),(2,3),(2,7)]
goal = 7
#%%
bees = [2]
smoke = [4,5,6]
G = nx.Graph()
G.add_edges_from(points_list)
mapping={0:'Start', 1:'1', 2:'2 - Bees', 3:'3', 4:'4 - Smoke', 5:'5 - Smoke', 6:'6 - Smoke', 7:'7 - Beehive'} 
H=nx.relabel_nodes(G,mapping)
pos = nx.spring_layout(H)
nx.draw_networkx_nodes(H,pos,node_size=[200,200,200,200,200,200,200,200])
nx.draw_networkx_edges(H,pos)
nx.draw_networkx_labels(H,pos)
plt.show()
#%%
mat = 8
r = np.matrix(np.ones(shape=(mat,mat)))
r *= -1
#%%
for point in points_list:
    print(point)
    if point[1] == goal:
        r[point] = 100
    else:
        r[point] = 0
        
    if point[0] == goal:
        r[point[::-1]] = 100
    else:
        # reverse of point
        r[point[::-1]] = 0
# add goal point round trip
r[goal,goal] = 100
r
#%%
Q = np.matrix(np.zeros([mat,mat]))
enviro_bees = np.matrix(np.zeros([mat,mat]))
enviro_smoke = np.matrix(np.zeros([mat,mat]))
#%%
# Learning parameter
gamma = 0.8

init_state = 1

enviro_matrix = enviro_bees - enviro_smoke

def available_actions(state):
    current_state_row = r[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

available_act = available_actions(init_state)

def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action

action = sample_next_action(available_act)

def collect_environmental_data(action):
    found = []
    if action in bees:
        found.append('b')
    if action in smoke:
        found.append('s')
    return (found)

def update(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index,size=1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    
    Q[current_state, action] = r[current_state, action] + gamma*max_value
    print('max_value',r[current_state,action] + gamma * max_value)
    
    environment = collect_environmental_data(action)
    if 'b' in environment:
        enviro_matrix[current_state, action] += 1
    if 's' in environment:
        enviro_matrix[current_state, action] += 1
    
    if (np.max(Q)>0):
        return(np.sum(Q/np.max(Q)*100))
    else:
        return (0)
update(init_state,action,gamma)

enviro_matrix_snap = enviro_matrix.copy()

def available_actions_with_enviro_help(state):
    current_state_row = r[state,]
    av_act = np.where(current_state_row >= 0)[1]
    #if there are multiple routes, penalize negative routes
    env_pos_row = enviro_matrix_snap[state,av_act]
    if (np.sum(env_pos_row < 0)):
        temp_av_act = av_act[np.array(env_pos_row)[0]>=0]
        if len(temp_av_act) > 0:
            print('going from:', av_act)
            print('to:', temp_av_act)
            av_act = temp_av_act
    return av_act
#%%
# Training
scores = []
for i in range(700):
    current_state = np.random.randint(0,int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    score = update(current_state,action,gamma)
    scores.append(score)
    print('Score:',str(score))
print('Trained Q Matrix:')
print (Q/np.max(Q)*100)

# Testing
current_state = 0
steps = [current_state]

while current_state !=7:
    next_step_index = np.where(Q[current_state,]==np.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index
print('Most efficient path:')
print(steps)

plt.plot(scores)
plt.show()