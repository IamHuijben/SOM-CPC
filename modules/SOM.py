
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import warnings

class GaussianNeighbourhood():
    def __init__(self, nr_nodes, n_epochs, sigma_end, omit_center_weight):
        """
        nr_nodes (int): Number of nodes in the SOM.
        n_epochs (int): Number of training epochs
        sigma_end (float): Standard deviation of the Gaussian kernel at the end of training.
        omit_center_weight (bool): If set to True, the center weight of 1 is set to zero, such that this weight belonging to the selected node can be applied separately in the commitment loss.
        """
        self.nr_nodes = nr_nodes
        self.som_dim_1d = int(np.sqrt(self.nr_nodes))
        
        self.n_epochs = n_epochs
        self.sigma_0 = np.sqrt(nr_nodes)//2 #Use the radius of the grid as start sigma
        self.sigma_end = sigma_end 
        
        # Compute the decay factor such that the sigma ends with sigma_end after n_epochs
        self.decay = -(n_epochs) / np.log(self.sigma_end/self.sigma_0)
        self.sigmas = self.sigma_0 * np.exp(-np.arange(n_epochs)/self.decay)

        self.omit_center_weight = omit_center_weight
        

    def convert_k_to_kx_ky(self, k):
        k_x = k // self.som_dim_1d
        k_y = k % self.som_dim_1d
        return torch.stack([k_x,k_y],-1) #[bs, 2]

    def compute_distances(self, selected_nodes_kx_ky):
        """
        selected_nodes_kx_ky (torch.Tensor): Contains the selected node indices for all elements in the batch. Shape: [bs, 2] 
        
        Returns:
        distance_grid (torch.Tensor): Grid with squared euclidean distances to each of the different nodes, from the selected node, for each element in the batch. Shape: [bs,sqrt(N),sqrt(N)]
        """
        
        bs = selected_nodes_kx_ky.shape[0]
        #selected_node_kx_ky = torch.randint(0,10,(bs, 2)) #[bs, 2 (k_x,k_y)]
        XX,YY = torch.meshgrid(torch.arange(self.som_dim_1d),torch.arange(self.som_dim_1d)) #ij indexing by default
        XX = XX.to(selected_nodes_kx_ky.device)
        YY = YY.to(selected_nodes_kx_ky.device)
        delta_x = XX.flatten().repeat((bs,1)) - selected_nodes_kx_ky[:,1].unsqueeze(-1).type(torch.float32)
        delta_y = YY.flatten().repeat((bs,1)) - selected_nodes_kx_ky[:,0].unsqueeze(-1).type(torch.float32)
        distance_grid = (torch.norm(torch.stack([delta_x,delta_y]), p=2, dim=0)**2)
        return distance_grid.reshape((-1,self.som_dim_1d, self.som_dim_1d))
        
    def compute_weighing(self, k, epoch):
        """
        k (torch.Tensor): Contains the selected node indices for all elements in the batch. Shape: [bs] 
        """
        selected_nodes_kx_ky = self.convert_k_to_kx_ky(k)
        distance_grid = self.compute_distances(selected_nodes_kx_ky)
        weighing = torch.exp(-distance_grid/(2*self.sigmas[epoch]**2)) #[bs, sqrt(N), sqrt(N)]

        if self.omit_center_weight:
            weighing[torch.abs(distance_grid) < 0.1] = 0. #Only works when distance is L2-squared.
        return weighing


class SOMQuantizer(nn.Module):
    def __init__(self, som_nodes=256, z_dim=128, transitions=False, gaussian_neighbourhood=None,**kwargs):
        # This is the SOM quantization module. It is similar to Vector Quantization, except that it also returns the neighbours 
        super(SOMQuantizer, self).__init__()       
        self.z_dim = z_dim  # the latent dimension 
        self.som_nodes = som_nodes  #the number of som nodes
        self.som_size_1d = np.uint16(np.sqrt(self.som_nodes))
        
        if gaussian_neighbourhood is not None:
            assert isinstance(gaussian_neighbourhood, dict)
            self.neighbourhood_inst = GaussianNeighbourhood(nr_nodes=som_nodes, n_epochs=gaussian_neighbourhood['n_epochs'], sigma_end=gaussian_neighbourhood.get('sigma_end',0.5),omit_center_weight=gaussian_neighbourhood.get('omit_center_weight',False))
        else:
            self.define_lookup_table_neighbours()

        self.embedding = nn.Embedding(self.som_nodes, self.z_dim) # create the embedding dictionary.
        self.embedding.weight.data.uniform_(-np.sqrt(1/self.z_dim), np.sqrt(1/self.z_dim)) 

        self.transitions = transitions
        if self.transitions:
            self.log_probs = nn.Parameter(torch.log(torch.ones(som_nodes,som_nodes) / som_nodes))
            
        else:
            self.log_probs = torch.zeros(1)
            
    def pairwise_distances(self, x, y=None):
        #From: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065 
        '''
        Input: x is a Nxd matrix # [1x64]
              y is an optional Mxd matirx [100x64]
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'. [100x1]
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)
        
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        return torch.clamp(dist, 0.0, np.inf)

    def define_lookup_table_neighbours(self):
        neighbour_lookup = torch.zeros((self.som_nodes, 4), dtype=torch.int64)

        #The SOM grid is an x-y coordinate system with the 0,0 in the left upper corner, walking downwards.
        for k in range(self.som_nodes):
            k_x = torch.as_tensor(k // self.som_size_1d)
            k_y = torch.as_tensor(k % self.som_size_1d)
            k = torch.as_tensor(k) 

            k_up = torch.where(k_y > 0,k-1,k) 
            k_down = torch.where(k_y<self.som_size_1d-1,k+1,k)
            k_left = torch.where(k_x >0,k-self.som_size_1d,k)
            k_right = torch.where(k_x<self.som_size_1d-1,k+self.som_size_1d,k)

            neighbour_lookup[k] = torch.stack([k_left, k_right, k_down, k_up])
        self.register_buffer('neighbour_lookup', neighbour_lookup) #In this way this lookup table is automatically placed to the same device when model is pushed to a device.

    def codebook_vec_selection(self, k):
        # k is one-hot vector of shape: [BS, nr_nodes]
        codebook = self.embedding.weight.unsqueeze(0).expand((k.shape[0],-1,-1)) # Make the shape from [nr_nodes , F] to [BS, nr_nodes, F] without copying the tensor
        return torch.sum(codebook*k.unsqueeze(-1),1) #[BS, F]

    def forward(self, z_e, epoch=None, **kwargs): 
        """ 
        z_e (tensor): Unquantized embedding of size bs x z_dim
        epoch (int): Training epoch

        Returns: 
        z_e (tensor): Continuous embedding of size bs x z_dim
        k (scalar): Index of the closest embedding in the dictionary. Shape: bs
        z_q (tensor): Quantized embedding  of size bs x z_dim
        codebook_idxs_neighbours (tensor): Index of the closest embedding in the  dictionary for all neighbours. Shape: bs x nr_neighbours
        neighbour_weighing (torch.tensor): Contains epoch-dependent weights for all nodes in the map when using gaussian kernel neighbourhood. 
        distance_matrix (tensor): Of shape bs x som_nodes
        """
        if len(z_e.shape) == 3:
            if z_e.shape[-1] == 1:
                z_e = z_e.squeeze(-1)
            else:
                z_e = torch.nn.AdaptiveAvgPool1d(output_size=1)(z_e).squeeze(-1)
                warnings.warn('The last dimension is reduced using adaptive avg pooling to have 1D feature vectors for the SOM.')

                
        codebook = self.embedding.weight #[n,z_dim]  

        # For each continuous embedding, compute the distance to all codebook vectors.
        distance_matrix = self.pairwise_distances(z_e,codebook) # output of size [bs x som_nodes]

        # Find the node index for which the codebook vector is closest to the continuous embedding.
        k = torch.argmin(distance_matrix,axis=-1) #[bs] 

        # Extract z_q from the codebook.
        z_q = self.codebook_vec_selection(one_hot(k,self.som_nodes))

        # Find the neighbours of z_q.
        if hasattr(self, 'neighbour_lookup'): #In case of Plus kernel
            codebook_idxs_neighbours = torch.index_select(self.neighbour_lookup, 0, k) #[BS x nr_neighbours]
            neighbour_weighing = None
        else:
            neighbour_weighing = self.neighbourhood_inst.compute_weighing(k, epoch=epoch)
            codebook_idxs_neighbours = None

        return dict(zip(self.return_output_var_names(),[z_e, k, z_q, codebook_idxs_neighbours, neighbour_weighing, distance_matrix]))     

    def return_output_var_names(self):
        return ['all_z_cont', 'all_codebook_idxs', 'all_z_disc', 'all_codebook_idxs_neighbours', 'neighbour_weighing', 'distance_matrix']




