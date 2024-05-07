# A Q-Learning Implementation for Warehouse Process Optimization

# Importing the libraries
import numpy as np

# Setting the parameters gamma and alpha for the Q-Learning
gamma = 0.75 # Discount factor
alpha = 0.9 # Learning rate

# PART 1 - DEFINING THE ENVIRONMENT

# Defining the states
location_to_state = {'A': 0,
                     'B': 1,
                     'C': 2,
                     'D': 3,
                     'E': 4,
                     'F': 5,
                     'G': 6,
                     'H': 7,
                     'I': 8,
                     'J': 9,
                     'K': 10,
                     'L': 11}

# Defining the actions
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# Defining the rewards
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])

# PART 2 - BUILDING THE AI SOLUTION WITH Q-LEARNING

# Making a mapping from the states to the locations
state_to_location = {state: location for location, state in location_to_state.items()}

# Making a function that returns the shortest route from a starting to ending location
def route_new(start, end):
    R_new = np.copy(R)
    end_index = location_to_state[end]
    R_new[end_index,end_index]=1000
    Q=np.array(np.zeros([12,12]))
    for i in range(1000):
        current_state = np.random.randint(0,12)
        actions = []
        for j in range(12):
            if(R_new[current_state,j]>0):
                actions.append(j)

        next_state = np.random.choice(actions)

        TD = R_new[current_state,next_state] + gamma*(Q[next_state,np.argmax(Q[next_state,])]) - Q[current_state,next_state]
        Q[current_state,next_state]=Q[current_state,next_state]+alpha*TD

    # print('Q_values:',Q.astype(int))
    route=[start]
    next_loc = start
    while(next_loc!=end):
        start_index=location_to_state[start]
        next_state = np.argmax(Q[start_index,])
        next_loc = state_to_location[next_state]
        route.append(next_loc)
        start = next_loc
    return route

# PART 3 - GOING INTO PRODUCTION

# Making the final function that returns the optimal route
def best_route(start , middle , end):
    return route_new(start,middle) + route_new(middle , end)[1:]

# Printing the final route
print('Route:')
print(best_route('E', 'K', 'G'))
