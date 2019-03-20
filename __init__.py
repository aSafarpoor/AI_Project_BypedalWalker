from gym.envs.box2d.lunar_lander import LunarLander
from gym.envs.box2d.lunar_lander import LunarLanderContinuous
from gym.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHardcore
from gym.envs.box2d.car_racing import CarRacing

import gym , random , time , json
from math import floor
import numpy as np

def hash_state(state):
    hstate = ""
     #print("    state : ",state[0:14])
    for i in state[0:14]:
        if i >= 1 :
            hstate += "10#"
        elif i <= -1:
            hstate += "-10#"
        else:
            hstate += str(int(i/0.2)) + "#"
    #print("          ",hstate)
    return hstate

def hash_action(action):
    global range_a
    print("     action : ",action)
    haction = ""
    for i in action:
        haction += str(floor(i * range_a)) + "#"
    print("          ",haction)
    return haction

def unhash_action(haction):
    global range_a
    l = list(map(int,haction.split("#")[:-1]))
    l_new = []
    for i in l:
        l_new.append(i / range_a)
    return l_new

def choose_action(state):
    global ptable , k , action_list ,rand_c ,amiss ,ahit
    add_state(state)
    if random.random() < rand_c or len(ptable[hash_state(state)]) == 0:
        index = str(random.randrange(len(action_list)))
        if not index in ptable[hash_state(state)].keys() :
            ptable[hash_state(state)][index] = 0
            amiss+=1
        else:
            ahit +=1
            #print("+==+===============+==+++++========+++=======")
        return unhash_action(action_list[int(index)])

    #f = u + k / n
    #print("kickASSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
    f = -float("inf") 
    max_action = -1
    for i in ptable[hash_state(state)].keys():
        if  f < ptable[hash_state(state)][i] : #+ (k / n_vals[i]):
            f = ptable[hash_state(state)][i] #+ (k / n_vals[i])
            max_action = i
    if f > -1:
        ahit +=1
        return unhash_action(action_list[int(max_action)])
    else :
        index = str(random.randrange(len(action_list)))
        if not index in ptable[hash_state(state)].keys() :
            ptable[hash_state(state)][index] = 0
            amiss+=1
        else:
            ahit+=1
        return unhash_action(action_list[int(index)])

def determine_state_value(state):
    global ptable , k , action_list ,rand_c
    add_state(state)
    if len(ptable[hash_state(state)]) == 0:
        return 0
    #f = u + k / n
    f = -float("inf")
    for i in ptable[hash_state(state)].keys():
        if  f < ptable[hash_state(state)][i] : #+ (k / n_vals[i]):
            f = ptable[hash_state(state)][i] #+ (k / n_vals[i])
    return f

def add_state(state):
    global ptable , action_list ,hit , miss
    hstate = hash_state(state)
    if not hstate in ptable.keys():
        ptable[hstate] = {}
        miss+=1
        #print("_____________________+++++++++++")
    else:
        hit+=1
  

def update_table(new_state,old_state,action,reward):
    global ptable ,action_list ,gama_c , alpha_c
    h_new_state = hash_state(new_state)
    h_old_state = hash_state(old_state)
    add_state(new_state)
    new_state_value = determine_state_value(new_state)
    action_index = str(action_list.index(hash_action(action)))
    #print("finalllllllllllllllllllllllold ",ptable[h_old_state][action_index])
    final_amount = ptable[h_old_state][action_index] * alpha_c + (1 - alpha_c) * (reward + gama_c * new_state_value)
    final_amount = int(final_amount * 100000) / 100000
    ptable[h_old_state][action_index] = final_amount
    #print("finalllllllllllllllllllllllnew ",ptable[h_old_state][action_index])

def choose_action2(s):
    global steps,total_reward,a,STAY_ON_ONE_LEG,PUT_OTHER_DOWN,PUSH_OFF,SPEED,state,moving_leg,supporting_leg,SUPPORT_KNEE_ANGLE,supporting_knee_angle , ptable
    # steps = 0
    # total_reward = 0
    # a = np.array([0.0, 0.0, 0.0, 0.0])
    # STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
    # SPEED = 0.29  # Will fall forward on higher speed
    # state = STAY_ON_ONE_LEG
    # moving_leg = 0
    # supporting_leg = 1 - moving_leg
    # SUPPORT_KNEE_ANGLE = +0.1
    # supporting_knee_angle = SUPPORT_KNEE_ANGLE
    
    #s, r, done, info = env.step(a)
    
    # total_reward += r
    # if steps % 20 == 0 or done:
    #     print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
    #     print("step {} total_reward {:+0.2f}".format(steps, total_reward))
    #     print("hull " + str(["{:+0.2f}".format(x) for x in s[0:4] ]))
    #     print("leg0 " + str(["{:+0.2f}".format(x) for x in s[4:9] ]))
    #     print("leg1 " + str(["{:+0.2f}".format(x) for x in s[9:14]]))
    #     print(total_reward)
    # steps += 1
    add_state(s)
    contact0 = s[8]
    contact1 = s[13]
    moving_s_base = 4 + 5*moving_leg
    supporting_s_base = 4 + 5*supporting_leg

    hip_targ  = [None,None]   # -0.8 .. +1.1
    knee_targ = [None,None]   # -0.6 .. +0.9
    hip_todo  = [0.0, 0.0]
    knee_todo = [0.0, 0.0]

    if state==STAY_ON_ONE_LEG:
        hip_targ[moving_leg]  = 1.1
        knee_targ[moving_leg] = -0.6
        supporting_knee_angle += 0.03
        if s[2] > SPEED: supporting_knee_angle += 0.03
        supporting_knee_angle = min( supporting_knee_angle, SUPPORT_KNEE_ANGLE )
        knee_targ[supporting_leg] = supporting_knee_angle
        if s[supporting_s_base+0] < 0.10: # supporting leg is behind
            state = PUT_OTHER_DOWN
    if state==PUT_OTHER_DOWN:
        hip_targ[moving_leg]  = +0.1
        knee_targ[moving_leg] = SUPPORT_KNEE_ANGLE
        knee_targ[supporting_leg] = supporting_knee_angle
        if s[moving_s_base+4]:
            state = PUSH_OFF
            supporting_knee_angle = min( s[moving_s_base+2], SUPPORT_KNEE_ANGLE )
    if state==PUSH_OFF:
        knee_targ[moving_leg] = supporting_knee_angle
        knee_targ[supporting_leg] = +1.0
        if s[supporting_s_base+2] > 0.88 or s[2] > 1.2*SPEED:
            state = STAY_ON_ONE_LEG
            moving_leg = 1 - moving_leg
            supporting_leg = 1 - moving_leg

    if hip_targ[0]: hip_todo[0] = 0.9*(hip_targ[0] - s[4]) - 0.25*s[5]
    if hip_targ[1]: hip_todo[1] = 0.9*(hip_targ[1] - s[9]) - 0.25*s[10]
    if knee_targ[0]: knee_todo[0] = 4.0*(knee_targ[0] - s[6])  - 0.25*s[7]
    if knee_targ[1]: knee_todo[1] = 4.0*(knee_targ[1] - s[11]) - 0.25*s[12]

    hip_todo[0] -= 0.9*(0-s[0]) - 1.5*s[1] # PID to keep head strait
    hip_todo[1] -= 0.9*(0-s[0]) - 1.5*s[1]
    knee_todo[0] -= 15.0*s[3]  # vertical speed, to damp oscillations
    knee_todo[1] -= 15.0*s[3]

    a[0] = hip_todo[0]
    a[1] = knee_todo[0]
    a[2] = hip_todo[1]
    a[3] = knee_todo[1]
    #print("a :",a)
    b = np.clip(0.5*a, -1.0, 0.9999)
    #print("b :",b)
    c = [b[0],b[1],b[2],b[3]]
    #print("c :",c)
    st_a = str(action_list.index(hash_action(c)))
    if not st_a in ptable[hash_state(s)].keys():
        ptable[hash_state(s)][st_a] = 0
    return c
    #env.render()
    #if done: break


######choose action 2
steps = 0
total_reward = 0
a = np.array([0.0, 0.0, 0.0, 0.0])
STAY_ON_ONE_LEG, PUT_OTHER_DOWN, PUSH_OFF = 1,2,3
SPEED = 0.29  # Will fall forward on higher speed
state = STAY_ON_ONE_LEG
moving_leg = 0
supporting_leg = 1 - moving_leg
SUPPORT_KNEE_ANGLE = +0.1
supporting_knee_angle = SUPPORT_KNEE_ANGLE
    

######end_choose action 2

env = gym.make('BipedalWalker-v2')
current_state = env.reset()
action = [0,0,0,0]
ptable = {}
try:
    with open("file.txt") as f:
        data = json.load(f)
    ptable = data['exDict']
except:
    print("file is empty")
k = 1000
gama_c = 0.95
alpha_c = 0.8
rand_c = 0.0

miss = 0
hit = 0

amiss = 0
ahit = 0

start_time = int(time.time())

print(len(ptable))

range_a = 5
action_list = []
for i in range (-range_a,range_a):
    for j in range(-range_a,range_a):
        for k in range (-range_a,range_a):
            for l in range(-range_a,range_a):
                action_list.append(hash_action([i/range_a,j/range_a,k/range_a,l/range_a]))

one_hour = 10 * 60 * 60
win_counter = 0
finish = start_time + one_hour
hour_counter = 0
while True :
    env.render()
    action = choose_action(current_state)
    #print(choose_action(current_state))
    #print(action)
    new_state, reward, done, info = env.step(action)
    #print(reward)
    """if abs(new_state[4] - new_state[9]) > 1.8:
        current_state = env.reset()
        continue"""
    #print(abs(new_state[4] - new_state[9]))
    update_table(new_state,current_state,action,reward)
    current_state = new_state
    if done :
        
        curr_time = time.time()
        if int((curr_time - start_time) / 3600) > hour_counter:
            hour_counter+= 1
            exDict = {'exDict': ptable}

            with open('file.txt', 'w') as file:
                file.write(json.dumps(exDict))
                file.close()
        print ("ptable :",len(ptable)," win : ",win_counter ," h : ",hour_counter, " chance : ",rand_c , " rew : ",reward)
        try:
            print("miss : ",miss,"hit : ",hit," hit ratio : ",hit/(miss+hit))
            print("amiss : ",amiss,"ahit : ",ahit," ahit ratio : ",ahit/(amiss+ahit))
        except Exception:
            pass
        """miss = 0
        hit = 0

        amiss = 0
        ahit = 0"""
        if time.time()  > finish :
            break
        current_state = env.reset()

    rand_c *= 0.9999992


# as requested in comment
exDict = {'exDict': ptable}

with open('file.txt', 'w') as file:
     file.write(json.dumps(exDict))
     file.close()
    
