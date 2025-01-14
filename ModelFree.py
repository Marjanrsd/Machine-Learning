import pandas as pd
import numpy as np
import shutil
import os, sys, time, datetime, json, random
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import ReLU
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

#5*5 maze can be coded as numpy array 7*15 - 0:occupied cell, 1:freecell
maze=np.array([
    [0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 1],
    [1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0]
    ])
#We've got occupied cells(obstacles), free cells, and target cell
visited_color = 0.6 #painted by grey 0.6
agent_color = 0.5 #painted by grey 0.5
target_color = 0.9
left = 0
up = 1
right = 2
down = 3

save_dir = os.path.join("/tmp/marjijoon_MF_2")
try:
    shutil.rmtree(save_dir)
except:
    print("oopsies, why did I get printed?")
os.makedirs(save_dir, exist_ok=True)

#actions
actions = {
    left:"left",
    right :"right",
    up :"up",
    down : "down"
}

action_num = len(actions) #4 actions
#exploration factor
exp = 0.1
#The qmaze class
class Qmaze(object):
    def __init__(self, maze, agent = (0,3)):
        self._maze = np.array(maze, dtype=np.float64)
        nrows, ncols = self._maze.shape
        #print(nrows-1, ncols-1)
        #print(self._maze[1])
        self.target = (14, 1) #target cell (last)
        self.closest = math.sqrt((agent[0]-self.target[0])**2 + (agent[1] - self.target[1])**2)
        self.free_cell = [(r,c) for r in range(nrows) 
                          for c in range(ncols)
                          if self._maze[r,c] == 1
                          ] #iterate through all cols and rows
                            #and get the cells that are free, i.e., x == 1 
        #print(self.free_cell[1])
        self.free_cell.remove(self.target)
        if self._maze[self.target] == 0:
            raise Exception("invalid: agent should be on the freecell!")
        self.reset(agent)

    def reset(self, agent):
        self.agent = agent
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row,col = agent # agent's current position
        self.maze[row, col] = agent_color #changing all the time
        self.state = (row,col, "start")
        print(self.state)
        self.min_reward = -0.4*self.maze.size #minimumreward threshold 
        print(self.maze.size)                 #results in losing the game
        self.total_reward = 0 # resetting everything                                  
        self.visited = set() #keep track of visited states without duplicates
        print(self.total_reward)

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = agent_row, agent_col, mode =self.state
        if self.maze[agent_row, agent_col]>0:
            self.visited.add((agent_row, agent_col)) #adding the agent position to the set
        valid_actions = self.valid_actions()
        if not valid_actions:
            nmode = "blocked"
        elif action in valid_actions:
            nmode = "valid"
            if action == left:
                ncol -= 1
            elif action == up:
                nrow -= 1
            elif action == right:
                ncol += 1
            elif action == down:
                nrow += 1
        else:
            mode = "invalid"
        self.state = (nrow, ncol, nmode) #new state
        print(self.state)
    
    def get_reward(self):
        agent_row, agent_col, mode = self.state
        nrows, ncols = self.maze.shape
        if agent_row == 14 and agent_col == 1: #we're at the target cell [-4,4]
            return 10 #our reward
        if mode == "blocked":
            return -0.7 # game over
        if (agent_row, agent_col) in self.visited:
            return -0.25 #if the agent went to the states that it previously explored punish them
        if mode == "invalid":
            return -0.70 #ur sitting on a state doing nothing!
        if mode == "valid":
            reward =  -0.05 #ur taking actions but u still get some negative default reward
            # distance_to_traget = math.sqrt((agent_row-14)**2 + (agent_col-1)**2)
            # if distance_to_traget < self.closest:
            #     reward += 0.25
            #     self.closest = distance_to_traget
            return reward  
        print(reward)

    def act(self, action):
        self.update_state(action) #call the update state function
        reward = self.get_reward() #call the get_reward function
        print(reward)
        self.total_reward += reward
        status = self.game_status() #call this function
        envstate = self.observe()
        return envstate, reward, status
        
    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1,-1)) #change it to one row an multiple cols. a vector.
        return envstate
    
    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        for row,col in self.visited:
            canvas[row,col] = visited_color
        agent_row, agent_col,_ = self.state
        canvas[agent_row, agent_col] = agent_color
        canvas[14,1] = target_color
        return canvas
        # for r in range(nrows):
        #     for c in range(ncols):
        #         if canvas[r,c] > 0:
        #             canvas[r,c] = 1
        # row, col, valid = self.state
        # canvas[row,col] = agent_color
        # return canvas
    
    def game_status(self):
        # if self.total_reward < self.min_reward:
        #     return "lose"
        agent_row, agent_col, mode = self.state
        nrows, ncols = self.maze.shape
        if agent_row == 14 and agent_col == 1:
            return "win"
        
        return "not_over"
    
    def valid_actions(self, cell = None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        
        if row == 0:
            actions.remove(1) #remove action one which is "up"
        elif row == nrows-1: #once in the target row
            actions.remove(3) #can't go to "down" when in the target state
        
        if col == 0:
            actions.remove(0) #can't go left
        elif col == ncols-1: #once in the target state
            actions.remove(2) #can't go right
        
        if row > 0  and self.maze[row-1,col] == 0: #if ur in a row and a row before ur current row is blocked (the same col)
            actions.remove(1) #u can't go up
        if row < nrows-1 and self.maze[row+1,col] == 0: #if ur not in the last row and the row after u is blocked
            actions.remove(3) # u can't go down

        if col > 0 and self.maze[row, col-1] == 0:
            actions.remove(0)
        if col < ncols-1 and self.maze[row,col+1] == 0:
            actions.remove(2)
        
        return actions
    
#Drawing the maze image
def show(qmaze):
    plt.grid ("on")
    nrows, ncols = qmaze.maze.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5,nrows,1)) #(x axis would be 0.5, 1.5, 2.5, 3.5, 4.5). 1 is the step size
    ax.set_yticks(np.arange(0.5,ncols,1))
    ax.set_xticklabels([])
    ax.set_yticklabels([]) #labels are empty lists for now
    canvas = np.copy(qmaze.maze)
    for row,col in qmaze.visited:
        canvas[row,col] = 0.6 #states visited are represented by this color
    agent_row, agent_col, _ = qmaze.state # _ variable is not going to be used
    canvas[agent_row, agent_col] = 0.3 # agent's current state is represented by this color
    print(agent_row, agent_col, nrows-1, ncols-1, 'asdfasdfasdf')
    canvas[14,1] = 0.9 #target state is represented by this color
    print(canvas.dtype)
    img = plt.imshow(canvas,interpolation = "none", cmap = "gray")
    return img
    

#make a function that takes a maze, the start cell of the agent, and a trained neural network
#that can calculate next action, as an argumennt 
def play_game(model, qmaze, agent_cell):
    qmaze.reset(agent_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate =  envstate
        #get next action
        q = model.predict(prev_envstate) 
        action = np.argmax(q[0])# finds the index of the maximum value in the vector
        #apply action, get reward and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == "win":
            return True
        elif game_status == "lose":
            return False
print("mARJAN")   
def completion_check(model, qmaze):
    for cell in qmaze.free_cell:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True
print("yessss")
#this is the class we collect our game episode in
class Experience(object):
    def __init__(self, model, max_memory = 100, discount = 0.95):
        self.model = model
        self.max_memory = max_memory #max length of episodes to keep
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]
    def remember(self, episode):
        #episode = [envstate, action, reward, enstate_next, game_over]
        #memory[i] = episode
        #envstate == 1-dimensional vector of the maze
        self.memory.append(episode)

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size = 10):
        env_size = self.memory[0][0].shape[1]#first element of the episode-extract the size of the second dimension
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
           envstate, action, reward, envstate_next, game_over = self.memory[j] 
           inputs[i] = envstate#image of the envi with all states
           targets[i] = self.predict(envstate)#q values
           Q_sa = np.max(self.predict(envstate_next))
           if game_over:
               targets[i, action] = reward
           else:
               targets[i,action] = reward + self.discount*Q_sa
        return inputs, targets

shortest_path = 22
#training neural network (our model)
def qtrain(model, maze, **opt): #opt means u can get different values from a dict as an argument
    global exp #exploration factor
    actions_taken_during_training = []  # To store actions taken during training
    performance_over_games = []
    steps_taken_per_game= []
    n_epoch = opt.get("n_epoch", 15000)#This line initializes the variable n_epoch by retrieving
    max_memory = opt.get("max_memory", 1000)  #the value associated with the key 'n_epoch' from the opt
    max_eps = opt.get("max_eps", 1000)
    data_size = opt.get("data_size", 50)      #dictionary. If the key is not present in opt, it defaults
    weights_file = opt.get("weights_file", "") #to the value 15000 
    name = opt.get("name", "model")            
    start_time = datetime.datetime.now()

    if weights_file: # if weights_file has a non-empty value
        print("loading weights from file: %s" %(weights_file,)) #weights are being loaded from a file
        model.load_weights(weights_file)
    
        #build an environment with numpy array maze
    qmaze = Qmaze(maze)
#initialize experience object
    experience = Experience(model, max_memory = max_memory)  

    win_history = [] #history of win/lose
    n_free_cells = len(qmaze.free_cell)   #number of freecells
    hsize = 5 #qmaze.maze.size//2 
    win_rate = 0 #ratio of successful outcomes to the total number of attempts or trials
    imctr = 1 #the number of iterations or steps in a loop or a sequence of operations

    for epoch in range(n_epoch):
        loss = 0
        agent_cell = [0,3] # random.choice(qmaze.free_cell)
        qmaze.reset(agent_cell)
        game_over = False
        #get 1 dimensional initial envstate
        envstate = qmaze.observe()
        #episode is one action
        n_episodes = 0
        #steps_in_current_game = 0
        
        print("wow")
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            #get next action
            if np.random.rand() < exp: #if this number is lower than our exploration threshold
                action = random.choice(valid_actions) #choose a random action
            else:
                action = np.argmax(experience.predict(prev_envstate)) #choose an action base on our policy

            #Apply action, get reward amd new envstate
            envstate, reward, game_status = qmaze.act(action)
            actions_taken_during_training.append(qmaze.state)
            
            #steps_in_current_game += 1
            # steps_taken_per_game.append(steps_in_current_game)
            print("Aha!")
            if game_status == "win":
                win_history.append(1)
                game_over = True
            elif game_status == "lose":
                win_history.append(0)
                game_over = True
            elif n_episodes > max_eps:
                game_over = True
            else:
                game_over = False #game is being played

            #store episode/experience
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1
            

            #Train neural network model
            inputs, targets = experience.get_data(data_size = data_size)
            h = model.fit(inputs, targets, epochs = 8, batch_size = 16, verbose = 0)
            loss = model.evaluate(inputs, targets, verbose = 0)

        this_game = len(actions_taken_during_training) - sum(steps_taken_per_game)
        steps_taken_per_game.append(this_game)
        #calculate performance and store it
        performance = this_game / shortest_path
        performance_over_games.append(performance)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize #calculating the win rate over the last hsize games

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))

        
        
        # if win_rate > 0.9 : #if the agent achieves a high win rate (greater than 90%), it reduces its exploration rate 
        #     exp = 0.5       #to focus more on exploiting the current knowledge
        if (np.mean(performance_over_games[-3:]) < 1.2):  #win_history keeps track of the recent outcomes of the episodes
            print("Reached 100%% win rate at epoch: %d" % (epoch,))
            break 
    
    #save trained model weights
    h5file = name + ".h5"
    #Constructs the filename for saving the model architecture in a JSON file. 
    #This JSON file will contain information about the model's architecture, such as the layers, activations, and connections
    json_file = name + ".json" 
    model.save_weights(h5file, overwrite = True)        
    with open(json_file, "w") as outfile:
        json.dump(model.to_json(), outfile) #Converts the model's architecture to a JSON-formatted string  
    end_time = datetime.datetime.now()
    dt = datetime.datetime.now() - start_time
    seconds = dt.total_seconds()
    t = format_time(seconds)
    print("files: %s, %s" % (h5file, json_file))
    print("n_epoch: %d, max_mem: %d, time: %s" % (epoch, max_memory, t))
    return actions_taken_during_training, steps_taken_per_game, performance_over_games

def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600
        return "%.2f hours" % (h,)

print("cooooollll")
def build_model(maze, lr = 0.001):
    model = Sequential()
    model.add(Dense(maze.size, input_shape = (maze.size,)))
    model.add(ReLU())
    model.add(Dense(maze.size))
    model.add(ReLU())
    model.add(Dense(action_num))
    model.compile(optimizer = "adam", loss = "mse")
    return model         

qmaze = Qmaze(maze)

model = build_model(maze)
vars = qtrain(model, maze, n_epoch=100, max_eps=100, max_memory=8*maze.size, data_size=32)
actions_during_training, steps_per_game, performances = vars
print("training completed")

#visualize
def visualize_game(qmaze, actions_taken):
    fig, ax = plt.subplots()
    # def update(frame):
    for fig_id, action in enumerate(actions_taken):
        agent_row, agent_col, _ = action
        ax.clear()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, qmaze.maze.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, qmaze.maze.shape[0], 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        canvas = np.copy(qmaze.maze)
        for row, col in qmaze.visited:
            canvas[row, col] = visited_color
        
        agent_row, agent_col, _ = actions_taken[fig_id]
        canvas[agent_row, agent_col] = agent_color
        canvas[14,1] = target_color

        img = plt.imshow(canvas, interpolation="none", cmap="gray")
        save_path = os.path.join(save_dir, f'{fig_id:06d}.jpg')
        plt.savefig(save_path)

    

visualize_game(qmaze, actions_during_training)

print("Done!")

def visualize_performance(performance):
    plt.figure()
    plt.plot(range(1, len(performance) + 1), performance) 
    plt.xlabel("Game Numbers")
    plt.ylabel("Performance(steps_taken/shortest path)")
    plt.title("Agent Performance Over Games")
    plt.show()

visualize_performance(performances)

def viz_state_max_q(qmaze):
    fig, ax = plt.subplots()
    ax.clear()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    max_qs = np.copy(qmaze.maze)
    experience = Experience(model)
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):      
            if canvas[r,c] != 0:
                saved_color = canvas[r,c]
                canvas[r,c] = agent_color # draw agent
                canvas[14,1] = target_color # draw goal
                qs = experience.predict(canvas.reshape(1,-1))
                max_qs[r,c] = float(np.max(qs))
                canvas[r,c] = saved_color
            else:
                max_qs[r,c] = -0.15
    img = plt.imshow(max_qs, interpolation="none", cmap="jet")
    plt.colorbar()
    save_path = os.path.join(save_dir, f'visualized_q_fx.jpg')
    plt.savefig(save_path)

viz_state_max_q(qmaze)
plt.show()



