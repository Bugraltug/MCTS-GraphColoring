#!/usr/bin/env python
# coding: utf-8

# # Monte Carlo Tree Search 1.5.13
# ## Date: 2018
# ## Creator: Buğra Altuğ
# 
# 
# ### Notes: 
# + Resign might not effect the nnet because without resign, it will learn what are the bad actions are. 
# 
# ### Updates:
# + Informations about "args" are added.
# + "check_resign" method is added. When given number of vertices cannot be colored, then mcts ends the game.
# + Args in the neural net class given as parameter.
# + Dirichlet is added.
# + Resign on # of empty vertices are now dynamically set to random nnet's average.
# 
# ### Upcoming:
# + parallelization could be made on episodes!
# + "Variance_viz" could be added to here.
# + "play_arena" method could be surely optimized by reconstructing the Node tree.
# + **First action of the MCTS is depricated**. Furtmermore it slows and expands uselessly the MCTS tree *Color_Count* times!! First color of the vertex does not matter!
# + DataQueue could be added (data of the last # of games could be fed to the nnet)!

# ## Imports:

# In[18]:


# NN imports.
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from copy import deepcopy

# MCTS imports.
import random as rnd
import numpy as np
from math import sqrt, log, ceil

# For calculating time.
import time

# If training data is wanted to export to an file.
import csv

# For redundant warnings. 
import warnings
warnings.filterwarnings("ignore")


# ## Configurations:

# In[19]:


# Arguments.
'''
test_mode:  If True, same initial nnet and randomness.

model_location: Location of the nnet. If exists, uses it. Else creates a model to that location.
graph_location: Mandatory location of the graph.

color_count: # of colors will be used to color the graph.

hidden_layer_rate: # of hidden layers with respect to graph size.
epoch: # of epochs when training the nnet.
learning_rate: Learning rate when training the nnet.
batch_size: # of batches when training the nnet.
l2_coef: L2 Regularization coefficent.

turn_threshold: Highest action will be selected when turn count exceeds "turn_threshold" else probabilistic. Will bias exploration rate.
resign_treshold: If given number of blank vertices exceeds, exit the game.
derive_data: New data will be created by swapping colors.

numIters: # of iterations. Which contains # of episodes, comparison, and training of nnet.
numEps: # of episode per iteration. Which contains # of simulations per turn (turn = vertex count, which depends to graph).
numEpisodeSims: # of monte carlo simulations per episode.
c_game: Monte Carlo exploration/exploitation coefficent in the main games. 

numCompEps: # of episodes in the comparison of new and old nnet.
numComparisonSims: # of monte carlo simulations per episode.
comparisonSimulationCoef: # of incrementation of "numComparisonSims" for every "numCompEps".
c_comparison: Monte Carlo exploration/exploitation coefficent in the nnet comparisons.
'''
args = {
    'test_mode': False, 
    'model_location': 'data/models/queen6_6.model',
    'graph_location': 'data/graphs/queen6_6.col', 
    'color_count': 7, 
    'hidden_layer_rate': 0.5,
    'epoch': 2,
    'learning_rate': 0.001,
    'batch_size': 1,
    'l2_coef':0.5, 
    'turn_threshold': 0.7,
    'derive_data': False,
    'numIters': 1, 
    'numEps': 5,
    'numEpisodeSims': 50, 
    'c_game': 4,
    'numCompEps':1,
    'numComparisonSims': 1,
    'comparisonSimulationCoef': 0,
    'c_comparison': 2
}


# In[20]:


# Device configuration.
if torch.cuda.is_available():
    print("CUDA is using.")
    device = torch.device('cuda')
    torch.cuda.synchronize()
    cuda = True
else:
    print("CUDA is not using.")
    device = torch.device('cpu')
    cuda = False

# Random seed.
if args["test_mode"]:
    rnd.seed(666)
    np.random.seed(666)
    torch.manual_seed(666)
    if cuda:
        torch.cuda.manual_seed_all(666)
        torch.backends.cudnn.deterministic=True


# ## Neural Network Dependency

# In[21]:


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ## Torch Neural Network

# In[22]:


class NeuralNet(nn.Module):
    def __init__(self, input_size: "int", hidden_size: "int", action_size: "int"):
        super(NeuralNet, self).__init__()
        self.action_size = action_size
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
 
        self.fc3 = nn.Linear(hidden_size, self.action_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        
    def getL2Regulazation(self, l2_coef: "float"):
        if cuda:
            l2_reg = torch.tensor(0.).cuda()
        else:
            l2_reg = torch.tensor(0.)
            
        for param in self.parameters():
                l2_reg += torch.norm(param) 
        loss = l2_coef * l2_reg 
        return loss
             
    def forward(self, data: "2d [board, pi] torch.Tensor"):
        out = self.fc1(data)
        out = self.relu(out)
        pi = self.fc3(out)
        v = self.fc4(out)
        return self.softmax(pi), self.tanh(v)
 
    def loss_pi(self, targets, outputs):
        return -torch.sum(targets*torch.log(outputs+1e-9))/targets.size()[0]
 
    def loss_v(self, targets, outputs):
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]
 
#**
    def train(self, trainingBoard: "3d torch.Tensor", trainingPi: "2d torch.Tensor", trainingV: "1d torch.Tensor", learning_rate_: "float", epoch_: "int", batch_size_: "int", l2_coef: "float"):
        print("="*40, "Network Condition", "="*40)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate_)
        for epoch in range(epoch_):
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            chunks = len(trainingBoard) // batch_size_
 
            for i in range(chunks):
                # Forward pass
                out_pi, out_v = self.forward(trainingBoard[i*batch_size_:(i+1)*batch_size_])
                l_pi = self.loss_pi(trainingPi[i*batch_size_:(i+1)*batch_size_], out_pi)
                l_v = self.loss_v(trainingV[i*batch_size_:(i+1)*batch_size_], out_v)
                total_loss = l_pi + l_v + self.getL2Regulazation(l2_coef)
                pi_losses.update(l_pi.item(), trainingBoard.size(0))
                v_losses.update(l_v.item(), trainingBoard.size(0))
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            print("Epoch:", epoch+1/epoch_)
            print("Pi, V Loss: ", pi_losses.avg, v_losses.avg)
        print("="*99)

    def saveModel(self, graph_location):  
        torch.save(self, graph_location)
        
    def printModel(self):
        for param in self.parameters():
            print("Neural Networks parameters are: ")
            print(param)   


# ## Graph Reader from file:

# In[23]:


class GraphReader:

    def __init__(self, file_path: "Path String"):
        self.file = open(file_path, 'r')

        self.edges = []

        self.sequence = []
        self.vertices = []

    def __del__(self):
        self.file.close()

    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices.append(vertex)
            self.sequence.append(-1)

    def add_edge(self, v1, v2):
        while len(self.edges) < v1:
            self.edges.append([])
        self.edges[v1 - 1].append(v2 - 1)

    def read(self):
        for edge in self.file:
            vertex_1 = int(edge.split()[0])
            vertex_2 = int(edge.split()[1])
            self.add_vertex(vertex_1)
            self.add_vertex(vertex_2)
            self.add_edge(vertex_1, vertex_2)
            self.add_edge(vertex_2, vertex_1)
        self.edges = np.array(self.edges)
        self.sequence = np.array(self.sequence)


# ## Game Board Structure:

# In[24]:


class ColoredGraph:
    def __init__(self, sequence: "1d Int Numpy Array", edges: "Int Numpy Matrix", first: "int" = 0, colored_count: "int" = 0):
        self.edges = edges
        self.sequence = sequence
        self.vertex_count = len(self.sequence)
        self.first = first
        self.colored_count = colored_count

    # Resets the graph.
    def reset(self):
        self.first = 0
        self.colored_count = 0
        self.sequence = [-1 for _ in range(self.vertex_count)]

    # Sets its sequence to the given graph.
    def similitude(self, graph: "ColoredGraph"):
        self.first = graph.first
        self.colored_count = graph.colored_count
        self.sequence = np.copy(graph.sequence)

    # Applies the given color to the graph.
    def update(self, color: "int"):
        self.sequence[self.first] = color

    # Checks if graph is ended or not.
    def ended(self) -> "bool":
        if self.first == self.vertex_count:
            return True
        else:
            return False

    # Checks if color is legal.
    def legal(self, color: "int") -> "bool":
        if color == 0:  # If no color then True
            return True
        else:
            adjacent_vertices = self.edges[self.first]
            for vertex in adjacent_vertices:
                if self.sequence[vertex] == color:
                    return False
            return True

    # If newly painted vertex is colored then increments the counter.
    def check_colored_count(self):
        if self.sequence[self.first] != 0:
            self.colored_count += 1

    # Calculates condition as currently how many colored divided by total.
    def condition(self) -> "float":
        return self.colored_count / self.vertex_count


# ## MCTS Tree Structure:

# In[25]:


class Node:
    def __init__(self, parent: "Node" = None, visits: "int" = 0, q: "int" = 0, p_val: "float" = 0, color: "int" = -1):
        self.parent = parent
        self.children = []
        self.visits = visits
        self.q = q
        self.color = color
        self.p_val = p_val
        
    # Checks if any children has the color.
    def exists(self, color: "int") -> "bool":
        for child in self.children:
            if child.color == color:
                return True
        return False
    
    # Expands with the given pi values or False with backup.
    def grow(self, pol: "1d Int Numpy Array or False", z: "float"):
        if pol is not False:
            for index, prob in enumerate(pol):
                if prob > 0:
                    color = index + 1
                    self.expand(prob, color).backup(z)
        else:
            self.expand(1.0, 0).backup(z)

    # Create a new child with the given color then return it.
    def expand(self, p_val: "float", color: "int") -> "Node":
        child = Node(parent=self, p_val=p_val, color=color)
        self.children.append(child)
        return child

    # Return most promising node. "U = Q(s, a) + P(s, a)* √(ln(N(sp, a))/(1+N(s,a)))"
    def best_child(self, c: "int") -> "Node":
        best = float("-inf")
        node = None
        for child in self.children:
            ucb = child.q + ( c * child.p_val * sqrt(log(self.visits) / (1 + child.visits)) )
            if ucb > best:
                best = ucb
                node = child
        return node
    # Update values up to parent. Fixed!
    def backup(self, val: "float"):
        tmp_visits = self.visits
        self.visits += 1
        
        self.q = ((tmp_visits * self.q) + val) / self.visits
        if self.parent is not None:
            self.parent.backup(val)


# ## Monte Carlo Model with NN:

# In[26]:


class MCTS:
    def __init__(self):
        self._nnet = None
        self._root = None
        self._colored_graph_sample = None
        self.iteration = None
        self.pi = None

    # Single iteration of MCTS.
    def simulate(self, c: "int", timer_:"GeneralTimer"):
        self.iteration += 1 
        node = self._root
        while not self._colored_graph_sample.ended():
            # New Leaf node(s).
            if not node.children:
                timer_.simulation.start_time()
                if cuda:
                    board = torch.Tensor([self._colored_graph_sample.sequence]).cuda()
                else:
                    board = torch.Tensor([self._colored_graph_sample.sequence])
                timer_.simulation.stop_time()
                
                timer_.network.start_time()
                pol, z = self._nnet.forward(board)
                timer_.network.stop_time()
                
                timer_.simulation.start_time()
                pol, z = (pol[0].data).cpu().numpy(), (z[0].data).cpu().numpy()
                pol = self.legalise_pi(self._colored_graph_sample, np.array(pol))  
                node.grow(pol, z)
                timer_.simulation.stop_time()
                break
            # Try to reach leaf node.
            else:
                timer_.simulation.start_time()
                node = node.best_child(c)
                self._colored_graph_sample.update(node.color)
                self._colored_graph_sample.check_colored_count()
                self._colored_graph_sample.first += 1
                timer_.simulation.stop_time()
    
    # Trains the NNet with the given data. ** 
    def reinforce(self, ds: "[[[Int], [float], float], ...] List", learning_rate_: "float", epoch_: "int", batch_size_: "int", l2_coef: "float"):
        b, pi, r = zip(*ds)
        if cuda:
            b, pi, r = torch.Tensor(b).cuda(), torch.Tensor(pi).cuda(), torch.Tensor(r).cuda()           
        else:
            b, pi, r = torch.Tensor(b), torch.Tensor(pi), torch.Tensor(r)
        self._nnet.train(b, pi, r, learning_rate_, epoch_, batch_size_, l2_coef) 

    # Resets the tree of mcts.
    def reset_tree(self):
        self.iteration = 0
        self._root = Node()
    
    # Set root as the given color and removes the non-child members of the mcts tree.
    def update_tree(self, color_: "int"):
        self.iteration = 0
        for child in self._root.children:
            if child.color == color_:
                self._root = child
                break
        self._root.parent = None
        
    # Creates a new game and tree. (New board.)
    def initiate_sample(self, graph_: "ColoredGraph"):
        self.reset_tree()
        self._colored_graph_sample = ColoredGraph(np.copy(graph_.sequence),
                                                  np.copy(graph_.edges), graph_.first, graph_.colored_count)

    # Copies the game's colored graph. (End of simulation.)
    def synchronize_sample(self, graph_: "ColoredGraph"):
        self._colored_graph_sample.similitude(graph_)
    
    # (End of turn.)
    def sync_sample_update_tree(self, graph_: "ColoredGraph", color_: "int"):
        self.update_tree(color_)
        self._colored_graph_sample.similitude(graph_)

    # Resets everything. (End of episode.)
    def reset_sync_tree(self):
        self.reset_tree()
        self._colored_graph_sample.reset()
    
    # Returns a copy of its own neural net.
    def return_nnet(self) -> "NeuralNet":
        return deepcopy(self._nnet)
    
    # Sets the given neural net to use in mcts.
    def set_nnet(self, nnet_: "NeuralNet"):
        self._nnet = deepcopy(nnet_)
    
    # Sets pi (color probabilities) without inapplicable and no-color.
    def calculate_legal_pi(self, graph_: "ColoredGraph", action_count_: "int"):
        color_visits = [float(0) for _ in range(action_count_)]
        for child in self._root.children:
            color_visits[child.color] = child.visits
        color_probabilities = [visit / float(sum(color_visits)) for visit in color_visits]
        del color_probabilities[0]  # No-color is removed from probabilities.
        self.pi = self.legalise_pi(graph_, np.array(color_probabilities))

    # Masks inapplicable moves on calculate_legal_pi. If there is no applicable moves then false.
    @staticmethod
    def legalise_pi(graph_: "ColoredGraph", pi: "1d Int Numpy Array") -> "bool or 1d Int Numpy Array":
        inapplicable_prob = 0
        summed_pi = sum(pi)
        for index, prob in enumerate(pi):
            color = index + 1
            if not graph_.legal(color):
                inapplicable_prob += prob
                pi[index] = 0
        applicable_prob = summed_pi - inapplicable_prob
        # If there is no applicable color or could not be done without no-color returns false.
        if applicable_prob == 0 or sum(pi) == 0:
            return False
        for i in range(len(pi)):  # Else normalize calculate_legal_pi.
            pi[i] = pi[i] / applicable_prob
        return pi


# ## Graph Coloring game:

# In[27]:


class ColoredGraphGame:
    def __init__(self, graph_location: "str"):
        _reader = GraphReader(graph_location)
        _reader.read()
        self._colored_graph = ColoredGraph(_reader.sequence, _reader.edges)
        self.turn = 0
        del _reader

    # Returns sequence of the colored graph.
    def board(self) -> "1d Int Numpy Array":
        return self._colored_graph.sequence

    # Applies the given color to main graph and Ends the current turn.
    def play_turn(self, color: "int"):
        self.turn += 1
        self._colored_graph.update(color)
        self._colored_graph.check_colored_count()
        self._colored_graph.first += 1

    # Returns whether the game is ended or not.
    def eog(self):
        return self._colored_graph.ended()
    
    # Returns game condition.
    def get_condition(self):
        return self._colored_graph.condition()
        
    # Start a new episode.
    def reset_game(self):
        self.turn = 0
        self._colored_graph.reset()

    # Returns the game graph.
    def graph(self) -> "ColoredGraph":
        return self._colored_graph
    
    # Returns vertex count.
    def board_size(self) -> "int":
        return self._colored_graph.vertex_count
    
    # Prints the board and condition of the game.
    def print_game_condition(self):
        print("Condition is: ", self.get_condition(), " , Board is: ", self.board())


# ## Episodic DataSet for communication between model and its nn:
# ### Note: Board, pi and r data for one episode (Ex: If colors are 1 and 2 with no-color, 0, k is 2).

# In[28]:


class ExpandableEpisodicDataSet:
    def __init__(self, episode: "int", k_: "int"):
        self._k = k_
        self.episode = episode
        self._data = []
        self._r = None
    
    # Add new board, pi pairs to the data.
    def insert(self, board: "1d Int Numpy Array", pi: "1d Int Numpy Array"):
        self._data.append([np.copy(board), np.copy(pi)])

    # From 1 to "k" (maximum-color) increment color.
    def next_color(self, iteration: "int", color: "int") -> "int":
        if color > 0:
            new_color = (color + iteration) % self._k
            if new_color == 0:
                new_color = self._k
            return new_color
        else:
            return color

    # Derive new board, pi data pairs for "k" times (color times) then add it to the data.
    def derive(self):
        size = len(self._data)
        for index in range(size):
            board, pi = self._data[index]
            for i in range(self._k - 1):
                new_board = []
                for color in board:
                    new_board.append(self.next_color(i + 1, color))
                new_pi = np.roll(pi, i + 1)
                self._data.append([np.array(new_board), new_pi])

    # Set "r" of the whole episodic data.
    def set_reward(self, r_: "float"):
        self._r = r_
        
    # Generate a complete (board, pi, r) (Use After the "expand" method).
    def finalize(self) -> "2d [[Int], [float], float] List":
        eds = []
        for b, pi in self._data:
            eds.append([b.tolist(), pi.tolist(), self._r])
        return eds
    
    # For exporting training data to an external file.
    def export_file(self, location: "str"):
        with open('{}episode_{}_dataset.txt'.format(location, self.episode), 'w') as outfile:
            for board, pi in self._data:
                outfile.write('{}, {}, {}\n'.format(board.tolist(), pi.tolist(), self._r))


# ## ColoredGraphGamePlayer:
# ### Note: It is bridge between ColoredGraphGame and Monte Carlo Model.

# In[29]:


class ColoredGraphGamePlayer:
    def __init__(self, graph_location: "str", color_count: "int"):
        self.episode = 0
        self.episode_cond = None
        self._color_count = color_count
        self._game = ColoredGraphGame(graph_location=graph_location)
        
        self._model = MCTS()
        self._model.initiate_sample(self._game.graph())
        
        _tmp_dirichlet = 10/((self._color_count)*self._game._colored_graph.vertex_count)
        self._dirichlet = Dirichlet(torch.tensor([_tmp_dirichlet for _ in range(self._color_count)])) 
        
        self._resign_treshold = float("inf")
        self._count_uncolored_vertices = 0
        
    # Sets its "_resign_treshold" parameter.
    def set_resign_treshold(self, treshold_: "int"):
        self._resign_treshold = treshold_
    
    # Returns the game board.
    def return_board(self) -> "1d Int Numpy Array":
        return self._game.board()
        
    # Train the nnet.
    def reinforce_nnet(self, data_set: "[[[Int List], [float List], float], ...] List", learning_rate_: "float", epoch_: "int", batch_size_: "int", l2_coef: "float"):
        self._model.reinforce(data_set, learning_rate_, epoch_, batch_size_, l2_coef)
        
    # Calculates the dirichlet probability on the pi.
    def calculate_dirichlet_probability(self, pi_: "float list") -> "float list":
        dirProb = np.float64([round(i.item(),3) for i in self._dirichlet.sample()])
        if sum(dirProb)!=1:
            indexMax = np.argmax(dirProb)
            dirProb[indexMax] = dirProb[indexMax] + 1.0 - sum(dirProb)
        return np.array([0.750*pi_[i] + 0.250*dirProb[i] for i in range(len(pi_))])
    
    # Prints episode's condition.
    def print_episode_conditions(self):
        print("Episode: ", self.episode, ", Condition is: ", self.episode_cond)
        
    # Change nnet of the model.
    def set_nnet(self, nnet_: "NeuralNet"):
        self._model.set_nnet(nnet_)
    
    # Return nnet of the model.
    def return_nnet(self) -> "NeuralNet":
        return self._model.return_nnet()
    
    # Changes the pi value to a more trainable pi and returns color.  
    def finalize_pi_color(self, pi_: "float List", turn_: "int", turn_threshold_: "int") -> "float list, int":
        if pi_ is False:
            color = 0
            pi = np.array([0 for _ in range(self._color_count)])
        else:
            T = (turn_ < self._game._colored_graph.vertex_count * turn_threshold_)
            if T:
                pi_ = self.calculate_dirichlet_probability(pi_)
                color = np.random.choice(len(pi_), p=pi_) + 1
                pi = pi_
            else:
                pi = [0 for _ in range(len(pi_))]
                max_index = np.argmax(pi_)
                pi[max_index] = 1
                color = max_index + 1
        return pi, color
    
    # Plays # of episode where in each episode, simulation count is increased by a coefficent. Finally returns average condition.
    def play_arena(self, episode_count_: "int", simulation_count: "int", simulation_coef: "float", c: "int", timer_: "GeneralTimer") -> "float":
        conditions_ = []
        for episode in range(episode_count_):
            while not self._game.eog():
                for _ in range(simulation_count + episode*simulation_coef):
                    self._model.simulate(c, timer_)
                    self._model.synchronize_sample(self._game.graph())

                # Predict the action.
                timer_.simulation.start_time()
                self._model.calculate_legal_pi(self._game.graph(), self._color_count + 1)

                if self._model.pi is not False:  
                    color = np.argmax(self._model.pi) + 1
                else:
                    color = 0
                self._game.play_turn(color)
                self._model.sync_sample_update_tree(self._game.graph(), color)
                timer_.simulation.stop_time()

            if self._game.get_condition() == float(1):
                print("Solution is found on the arena, Simulation count was", simulation_count + episode*simulation_coef, end=". ")
                return self._game.get_condition()
            else:
                conditions_.append(self._game.get_condition())
                self.reset_tree_game()
        return (sum(conditions_) / float(len(conditions_)))
    
    # Plays episode (set of turns).
    def play_game(self, simulation_count: "int", turn_threshold_: "int", c: "int", derive_data: "bool", timer_: "GeneralTimer") -> "2d [[Int], [float], float] List or bool":
        self._count_uncolored_vertices = 0
        episodic_data = ExpandableEpisodicDataSet(self.episode, self._color_count)
        while not self._game.eog():
            for _ in range(simulation_count):
                self._model.simulate(c, timer_)
                self._model.synchronize_sample(self._game.graph())
                
            # Predict the action.
            timer_.simulation.start_time()
            self._model.calculate_legal_pi(self._game.graph(), self._color_count + 1)
            pi, color = self.finalize_pi_color(self._model.pi, self._game.turn, turn_threshold_)
            
            if self.check_resign(color):
                timer_.simulation.stop_time()
                del episodic_data
                return False
            else:
                episodic_data.insert(self._game.board(), pi)
                self._game.play_turn(color)
                self._model.sync_sample_update_tree(self._game.graph(), color)
                timer_.simulation.stop_time()
            
        episodic_data.set_reward(self._game.graph().condition())
        self.episode_cond = self._game.graph().condition()
        if derive_data:
            episodic_data.derive()
        self.episode += 1
        return episodic_data.finalize()
    
    # If uncolored vertex count exceeds the treshold_, then resign.
    def check_resign(self, color_: "int") -> "bool":
        if color_ == 0:
                 self._count_uncolored_vertices += 1
        if self._count_uncolored_vertices > self._resign_treshold:
            return True
        else:
            return False
    
    # End of episode or comparison.
    def reset_tree_game(self):
        self._game.reset_game()
        self._model.reset_sync_tree()
    
    # Compare the nnet with model's nnet with # of episodes each increases # of simulations. 
    # If solution is found then returns True. 
    # Called on the end of one iteration (episode set).
    def set_best_nnet(self, old_nnet_: "NeuralNet", episode_count_: "int", simulation_count_: "int", simulation_coef_: "int", c_: "int", timer_: "GeneralTimer") -> "bool":
        self.episode = 0
        self.episode_cond = 0
        new_nnet_condition = self.play_arena(episode_count_, simulation_count_ , simulation_coef_, c_, timer_)
        if new_nnet_condition == float(1):
            print("Trained Network")
            return True
        
        new_nnet_ = self.return_nnet()
        self.set_nnet(old_nnet_)
        
        old_nnet_condition = self.play_arena(episode_count_, simulation_count_ , simulation_coef_, c_, timer_)
        if old_nnet_condition == float(1):
            print("Old Network")
            return True
        
        print("Trained Network's average condition is: ", new_nnet_condition, "on", episode_count_, "games.")
        print("Old Network's average condition is: ", old_nnet_condition, "on", episode_count_, "games.")
        if new_nnet_condition >= old_nnet_condition:
            self.set_nnet(new_nnet_)
            print("Trained Network is Selected!")
        else:
            print("Old Network is selected.")
        print("="*99)
        return False
        


# ## Coach:
# ### Note: Coach determines whether nnet is improved or not (Plays episodes).

# In[54]:


class Coach:
    def __init__(self, graph_location: "str", color_count: "int", hidden_layer_rate: "float", test_mode: "bool", model_location: "str" = ""):
        self._iteration = 0
        self.solution = None
         
        self._iteration_max_cond = 0
        self._iteration_total_cond = 0
        self._iteration_episode_count = None
         
        self._data_set = []
        self._game_player = ColoredGraphGamePlayer(graph_location=graph_location, color_count=color_count)
        if test_mode:
            try:       
                self._recorded_nnet = torch.load(model_location)
            except:
                self._recorded_nnet = NeuralNet(input_size=self._game_player._game.board_size(), hidden_size=ceil(self._game_player._game.board_size()*hidden_layer_rate), action_size=color_count).to(device)
                self._recorded_nnet.saveModel(model_location)
        else:
            self._recorded_nnet = NeuralNet(input_size=self._game_player._game.board_size(), hidden_size=ceil(self._game_player._game.board_size()*hidden_layer_rate), action_size=color_count).to(device)
        self._game_player.set_nnet(self._recorded_nnet)

    # Sets game player's resign treshold to the previous iterations uncolored vertex average. **
    def init_player_resign_treshold(self):
        average = self._iteration_total_cond / self._iteration_episode_count
        total_colored = round(average * self._game_player._game._colored_graph.vertex_count)
        treshold = self._game_player._game._colored_graph.vertex_count - total_colored
        print("="*42, "Resign Treshold", "="*42)
        print("Uncolored vertex count:", treshold)
        print("Will be higher or equal to:", average)
        print("="*99)
        self._game_player.set_resign_treshold(treshold)

    # In # of episodes: Play's game. If solution is found return else train the model's nnet.
    def iterate(self, episodes: "int", simulation_count: "int", turn_threshold_: "int", c: "int", learning_rate_: "float", epoch_: "int", batch_size_: "int", l2_coef: "float", derive_data: "bool", timer_: "GeneralTimer") -> "bool":
        self._iteration += 1
        self._iteration_episode_count = episodes
        while self._game_player.episode < episodes:
            episode_data = self._game_player.play_game(simulation_count, turn_threshold_, c, derive_data=derive_data, timer_=timer_)
            if episode_data: # Not resigned.
                self._data_set.extend(episode_data)
                self.update_iterate_conditions(self._game_player.episode_cond)
                # If a solution is found on the end of episode, then return True.
                if self._iteration_max_cond == float(1):
                    self.solution = self._game_player.return_board()
                    return True
            self._game_player.reset_tree_game()
        rnd.shuffle(self._data_set)
        timer_.network.start_time()
        self._game_player.reinforce_nnet(self._data_set, learning_rate_, epoch_, batch_size_, l2_coef)
        timer_.network.stop_time()
        self._data_set = []
        return False
     
    # Prints the avg, max conditions of a set of episodes (one iteration).
    def print_iterate_conditions(self):
        print("="*40, "Iteration", self._iteration, "="*40)
        print("Average of Episodes:", self._iteration_total_cond / self._iteration_episode_count)
        print("Maximum of Episodes:", self._iteration_max_cond)
         
    # Updates the condition values (avg, max) after each episode ends.
    def update_iterate_conditions(self, episode_condition: "float"):
        self._iteration_total_cond += episode_condition
        if self._iteration_max_cond < episode_condition:
            self._iteration_max_cond = episode_condition
     
    # Resets the gained avg, max conditions of a set of episodes.
    def reset_iterate_conditions(self):
        self._iteration_max_cond = 0
        self._iteration_total_cond = 0
     
    # Compares the model's nnet with "_recorded_nnet". End of one iteration. If solution is found, returns True.
    def set_best_nnet(self, episode_count_: "int", simulation_count_: "int", simulation_coef_: "int", c_: "int",  timer_: "GeneralTimer") -> "bool":
        self.reset_iterate_conditions()
        if self._game_player.set_best_nnet(self._recorded_nnet, episode_count_, simulation_count_, simulation_coef_, c_,  timer_):
            self.solution = self._game_player.return_board()
            return True
        self._recorded_nnet = self._game_player.return_nnet()
        return False


# ## Timer:

# In[55]:


class Timer:
    def __init__(self):
        self.total_time = 0
        self.tmp_time = 0
        
    # Start timer to calculate run time.
    def start_time(self):
        self.tmp_time = time.process_time()
    
    # Stop timer to cumulate run time (In seconds).
    def stop_time(self):
        self.total_time += time.process_time() - self.tmp_time
        
    # Return calculated time as minutes.
    def minutes(self) -> "int":
        return self.total_time/60
        
class  GeneralTimer:
    def __init__(self):
        self.total = Timer()
        self.network = Timer()
        self.simulation = Timer()
        self.preprocess = Timer()
        
    # Calculates the trivial percentage.
    @staticmethod
    def calculate_percentage(time: "int", base: "int") -> "int":
        return round((time / base)*100, 2)
    
    # Show general information about calculated run times.
    def display_info(self):
        print("="*40, "Time Informations", "="*40)
        print("Total Time:", round(self.total.minutes(), 3), "mins")
        print("Network Time:", round(self.network.minutes(), 3), "mins")
        print("Simulation Time:", round(self.simulation.minutes(), 3), "mins")
        print("Pre Process Time:", round(self.preprocess.minutes(), 3), "mins")
        
        
        percNet = str(self.calculate_percentage(self.network.minutes(), self.total.minutes()))
        percSim = str(self.calculate_percentage(self.simulation.minutes(), self.total.minutes()))
        percPro = str(self.calculate_percentage(self.preprocess.minutes(), self.total.minutes()))
        print("%'s are: Network is "+percNet+"%, Simulation is "+percSim+"%, Pre Process is "+percPro+"%.")
        print("="*99)


# ## Validator (to check conflicts):

# In[56]:


class Validator:
    def __init__(self, file_path: "Path String", sequence: "1d Int Numpy Array"):
        self.file = open(file_path, 'r')
        self.sequence = sequence
        self.conflict_count = 0

    def __del__(self):
        self.file.close()

    def iterate_all(self):
        for edge in self.file:
            if self.check_conflict(edge):
                self.conflict_count += 1

    # One line check for the vertex pairs.
    def check_conflict(self, edge: "String like: xx - xx"):
        vertex_1_index = int(edge.split()[0]) - 1
        vertex_2_index = int(edge.split()[1]) - 1
        if self.sequence[vertex_1_index] == 0:
            return False
        elif self.sequence[vertex_2_index] == 0:
            return False
        elif self.sequence[vertex_1_index] == self.sequence[vertex_2_index]:
            return True
        else:
            return False


# ## Main:

# In[57]:


timer = GeneralTimer()
timer.total.start_time()

coach = Coach(graph_location=args["graph_location"], color_count=args["color_count"], hidden_layer_rate=args["hidden_layer_rate"], test_mode=args["test_mode"], model_location=args["model_location"])

coach.iterate(episodes=args["numEps"], simulation_count=args["numEpisodeSims"], turn_threshold_=args["turn_threshold"], c=args["c_game"], learning_rate_=args["learning_rate"], epoch_=args["epoch"], batch_size_=args["batch_size"], l2_coef=args["l2_coef"], derive_data=args["derive_data"], timer_=timer)
coach.init_player_resign_treshold()
coach.print_iterate_conditions()
coach.set_best_nnet(episode_count_=args["numCompEps"], simulation_count_=args["numComparisonSims"], simulation_coef_=args["comparisonSimulationCoef"], c_=args["c_comparison"], timer_=timer)

while coach._iteration < args["numIters"]:
    if coach.iterate(episodes=args["numEps"], simulation_count=args["numEpisodeSims"], turn_threshold_=args["turn_threshold"], c=args["c_game"], learning_rate_=args["learning_rate"], epoch_=args["epoch"], batch_size_=args["batch_size"], l2_coef=args["l2_coef"], derive_data=args["derive_data"], timer_=timer):
        break
    coach.print_iterate_conditions()
    if coach.set_best_nnet(episode_count_=args["numCompEps"], simulation_count_=args["numComparisonSims"], simulation_coef_=args["comparisonSimulationCoef"], c_=args["c_comparison"], timer_=timer):
        break
        
timer.total.stop_time()
timer.display_info()
print("="*40, "Arguments", "="*40)
print(args)
print("="*99)
if coach.solution is not None:
    print("Solution is found on the",coach._iteration,"!")
    validator = Validator(args["graph_location"], coach.solution)
    validator.iterate_all()
    print("Total conflicts are: ", validator.conflict_count, ".")
    del validator
else:
    print("Solution couldn't found!")
print("="*99)


# ## Tests:
