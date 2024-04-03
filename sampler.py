from geometric_network import Network_Model
import numpy as np
import scipy.stats
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import itertools as it

def init_positions(A):
    # Initialise positions baseed on eigenvectors
    D = np.diag(np.sum(A,axis=1)**(-0.5))
    evals,eigen = np.linalg.eigh(D @ A @ D)
    unscaled = eigen[:,-1]
    positions = 2*np.pi*scipy.stats.rankdata(unscaled)/len(unscaled)
    return positions
    
def joinup(arr):
    return np.array(list(it.chain(*arr)))

def sort_nodes_by_component(A,positions):
    G = nx.from_numpy_array(A)
    components = list(nx.connected_components(G))
    components = [list(component) for component in components]
    adjacency_blocks = [nx.adjacency_matrix(G.subgraph(component),
                                            nodelist=component
                                           ).todense() 
                        for component in components]

    true_positions = joinup([positions[np.array(component)] for component in components])
    A_true = scipy.linalg.block_diag(*adjacency_blocks)
    
    start_positions = joinup([init_positions(A) for A in adjacency_blocks])
    
    return A_true,true_positions,start_positions


def reflecting(x,T):
    
    if np.isinf(T):
        if x < 0:
            return -x
    
    x = x % (2*T)
    if 0 <= x < T:
        return x
    elif T <= x < 2*T:
        return T - (x - T)
    else:
        print(x,T)
    raise IndexError

class Sampler:
    
    def __init__(self, A):
        
        self.A = A
        
        self.n_nodes = A.shape[0]
        self.noise = 0.01
        self.steps_elapsed = 0
        
        self.nlogp_history = []
        
        self.current = Network_Model(self.n_nodes)
        self.test = Network_Model(self.n_nodes)
        
        #self.positions
    def record(self, nlogp_test, extra):
        temp = [extra, nlogp_test, *self.current.params, *self.positions]
        self.nlogp_history.append(temp)
    
    
    def sample_params(self,noise):
        # select to a new position to potentially jump to
        test_c_ = reflecting(self.current.c_ + noise*np.random.randn(), 1.0)
        test_r_ = reflecting(self.current.r_ + noise*np.random.randn(), 0.5)
        test_lambda_ = reflecting(self.current.lambda_ + noise*np.random.randn(), np.inf)
        test_params = [test_lambda_, test_c_, test_r_]
        self.test.set_params(test_params)
        
    def sample_positions(self,noise):
        self.test_positions=np.array([reflecting(x + noise*np.random.randn(), 2*np.pi) 
                                      for x in self.positions])

    def update_params(self,beta):

        nlogp_old = self.current.nlogp(self.A,self.positions)
        nlogp_test = self.test.nlogp(self.A,self.positions)

        # should be *smaller* when the new position is higher probability
        # i.e, when -lnp(test) is small compared to -lnp(previous)

        nlogp_diff =  nlogp_test - nlogp_old
        p = np.exp(-beta*nlogp_diff)

        if p > np.random.rand():
            #self.nlogp_history.append([self.test.params, nlogp_test, "par"])
            self.current = self.test
            self.record(nlogp_test,"params")

        

    def update_positions(self,beta):

        nlogp_old  = self.current.nlogp(self.A,self.positions)
        nlogp_test = self.current.nlogp(self.A,self.test_positions)

        # should be *smaller* when the new position is higher probability
        # i.e, when -lnp(test) is small compared to -lnp(previous)
        nlogp_diff =  nlogp_test - nlogp_old
        p = np.exp(-beta*nlogp_diff)

        if p > np.random.rand():
            
            self.positions = self.test_positions
            self.record(nlogp_test,"position")



if __name__ == "__main__":

    # Initialise a ground truth msodel and draw a ground truth network from it       
    ground_truth = Network_Model(100)
    lambda_true,c_true,r_true  = 0.1, 0.6, 0.4
    ground_truth.set_params([lambda_true,c_true,r_true])
    positions = ground_truth.sample_position()
    A = ground_truth.sample_network(positions)

    # Innitalise and estimate using spectra
    A_true,true_positions,start_positions = sort_nodes_by_component(A,positions)
    sampler = Sampler(A_true)
    sampler.positions = start_positions
    sampler.current.set_params([0.1,0.1,0.1])

    # run the sampler
    n_steps = 500000
    for i in tqdm.trange(n_steps):
        inv_temp = 10*np.sqrt(i/n_steps)

        noise = 0.1/(1+(100*np.sqrt(i/n_steps)))

        sampler.sample_params(noise)
        sampler.sample_positions(4.0*noise)
        sampler.update_params(inv_temp)
        sampler.update_positions(inv_temp)
    df = pd.DataFrame(sampler.nlogp_history)

    import uuid
    folder = "sampling/"+str(uuid.uuid4())

    import os
    os.makedirs(folder, exist_ok=True)
    df.to_csv(folder+"/history.csv")
    pd.Series(ground_truth.params).to_csv(folder+"/true_params.csv")
    pd.Series(true_positions).to_csv(folder+"/true_positions.csv")
    pd.DataFrame(A_true).to_csv(folder+"/true_adjacency.csv")


