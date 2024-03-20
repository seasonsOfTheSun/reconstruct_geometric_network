
import numpy as np
import scipy.stats
import scipy.special
import networkx as nx
import matplotlib.pyplot as plt

class Geometric_Network:

    def __init__(self,n_nodes):
        self.n_nodes = n_nodes
        
    def nlogp_interaction(self,positions):
        # matrix of interaction probabilities given positions
        distances = np.abs((positions.reshape(-1,1) - positions.reshape(1,-1)))
        return self.nlog_connection_p(distances)

    def p_interaction(self, positions):
        nlogp_interaction_values = self.nlogp_interaction(positions)
        p_interaction = np.exp(-nlogp_interaction_values)
        return p_interaction

    def nlog_network(self,A,nlogp_interaction):
        # probability of observing network

        dodge = 0.0001

        connected = nlogp_interaction*A
        A_complement = (np.ones(self.n_nodes)-np.eye(self.n_nodes) - A)
        p_interaction = np.exp(-nlogp_interaction)
        not_connnected = -np.log(1-p_interaction+dodge)*A_complement

        return np.sum(connected+not_connnected)

    def nlogp(self,A,positions):
        nlogp_position = np.sum(self.nlog_position(positions))
        nlogp_interaction_values = self.nlogp_interaction(positions)
        nlogp_network = self.nlog_network(A,nlogp_interaction_values)
        return nlogp_network + nlogp_position

    def sample_network(self,positions):
        randomness = np.random.rand(self.n_nodes,self.n_nodes)
        return  randomness < self.p_interaction(positions)





class Interval_Beta(Geometric_Network):

    def __init__(self,n_nodes):
        super().__init__(n_nodes)

    def set_params(self,params):
        
        self.params = params
        a,b,sigma = params
        self.a = a
        self.b = b
        self.sigma = sigma
        

    # position probabillity distribution
    def position_hamiltonian(self,x):
        return np.log(x)*(self.a-1) * np.log(1-x)*(self.b-1)

    def position_logpartition(self):
        return (np.log(scipy.special.gamma(self.b))+np.log(scipy.special.gamma(self.a))
                - np.log(scipy.special.gamma(self.a+self.b)))

    def nlog_position(self,x):
        return self.position_hamiltonian(x) + self.position_logpartition()

    def nlog_connection_p(self,distance):
        # probability of connection given distances
        return 0.5*(((distance)/self.sigma)**2)

    def sample_position(self):
        return scipy.stats.beta(self.a, self.b).rvs(self.n_nodes)





class Network_Model(Geometric_Network):

    def __init__(self,n_nodes):
        super().__init__(n_nodes)

    def set_params(self,params):
        
        self.params = params
        lambda_,c_,r_ = params
        self.lambda_ = lambda_
        self.c_ = c_
        self.r_ = r_
        

    # position probabillity distribution
    def position_hamiltonian(self,x):
        return self.lambda_*np.mod(x, 2*np.pi)

    def position_logpartition(self):
        return np.log(1 - np.exp(-self.lambda_*2*np.pi)) - np.log(self.lambda_)

    def nlog_position(self,x):
        return self.position_hamiltonian(x) + self.position_logpartition()

    def nlog_connection_p(self,distance,dodge=0.00001):
        # probability of connection given distances
        return -np.log(self.c_ * (self.r_ - np.minimum(distance, self.r_))/self.r_+dodge)
        
    def clip_position(self, x, dodge=1e-3):
          x = np.maximum(x,dodge)
          x = np.minimum(x,2*np.pi-dodge)
    


    def sample_position(self):
        return scipy.stats.expon(scale=1/self.lambda_).rvs(self.n_nodes)%(2*np.pi)
        







