import torch
import torch.nn as nn
import math
import torch.nn.functional as F 

class commitment_loss(nn.Module):
    def __init__(self, reduction='sum', similarity_metric='mse', detach_from_encoder=False, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.similarity_metric = similarity_metric.lower()
        self.detach_from_encoder = detach_from_encoder

    def convert_idxs_to_codebook_vec(self, codebook, idxs):
        # If idxs is a multi-dimensional array, just consider all elements separately and return the corresponding codebook vectors. 
        flat_idxs = idxs.flatten()
        feature_size = codebook.shape[-1]
        return codebook[flat_idxs].view(*idxs.shape+(feature_size,)).squeeze()

    def compute_loss(self, z_disc, z_cont):
        if self.similarity_metric == 'mse':
            if self.detach_from_encoder:
                loss = F.mse_loss(z_disc,z_cont.detach(),reduction='none')    
            else:    
                loss = F.mse_loss(z_disc,z_cont,reduction='none')        
            loss = torch.mean(loss, dim=-1, keepdim=False) #[bs]
        elif self.similarity_metric == 'mae':
            if self.detach_from_encoder:
                loss = F.l1_loss(z_disc,z_cont.detach(),reduction='none')    
            else:    
                loss = F.l1_loss(z_disc,z_cont,reduction='none')        
            loss = torch.mean(loss, dim=-1, keepdim=False) #[bs]
        elif self.similarity_metric == 'dot':
            if self.detach_from_encoder:
                loss = torch.matmul(z_cont.detach(),z_disc)    
            else:
                loss = torch.matmul(z_cont,z_disc)    
        return loss

    def forward(self, pred, **kwargs):
        """
        Args:
            pred (list): List of model outputs: [k, z_e, z_q, z_q_neighbours, distance_matrix, all_embeddings, log_probs]
        """
        if len(pred) == 1 and list(pred)[0] == 'out': #sequential model
            pred = pred['out']

        if 'all_z_disc' in pred:
            all_z_cont, all_z_disc =  pred['all_z_cont'], pred['all_z_disc']
        else:
            all_z_cont, idxs =  pred['all_z_cont'], pred['all_codebook_idxs']
            idxs = idxs[-1] if isinstance(idxs,list) else idxs
            all_z_disc = self.convert_idxs_to_codebook_vec(kwargs['named_parameters']['quantizer.embedding.weight'],idxs)

        z_cont = all_z_cont[-1] if isinstance(all_z_cont,list) else all_z_cont
        z_disc = all_z_disc[-1] if isinstance(all_z_disc,list) else all_z_disc

        loss = self.compute_loss(z_disc, z_cont)
        assert len(loss.shape) == 1
        if self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            raise NotImplementedError        
        return loss, {} # return an empty dictionary for extra loss information.
    
class SOM_loss(nn.Module):
    def __init__(self, reduction='sum',similarity_metric='mse', detach_from_encoder=True, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.similarity_metric = similarity_metric.lower()
        self.detach_from_encoder = detach_from_encoder

    def convert_idxs_to_codebook_vec(self, codebook, idxs):
        # If idxs is a multi-dimensional array, just consider all elements separately and return the corresponding codebook vectors. 
        flat_idxs = idxs.flatten()
        feature_size = codebook.shape[-1]
        return codebook[flat_idxs].view(*idxs.shape+(feature_size,)).squeeze()

    def prepare_all_preds(self, pred, **kwargs):
        if len(pred) == 1 and list(pred)[0] == 'out': #sequential model
            pred = pred['out']

        neighbour_weighing = None
        if 'z_disc_neighbours' in pred:
            all_z_cont, z_disc_neighbours =  pred['all_z_cont'], pred['z_disc_neighbours']
            z_disc_neighbours = z_disc_neighbours[-1] if isinstance(z_disc_neighbours,list) else z_disc_neighbours

        elif 'all_codebook_idxs_neighbours' in pred and (pred['all_codebook_idxs_neighbours'] is not None and pred['all_codebook_idxs_neighbours'] != [None]): # In case of SOM_CPC or SOM-VAE
            all_z_cont, idx_neighbours =  pred['all_z_cont'], pred['all_codebook_idxs_neighbours'] 
            idx_neighbours = idx_neighbours[-1] if isinstance(idx_neighbours,list) else idx_neighbours
            z_disc_neighbours = self.convert_idxs_to_codebook_vec(kwargs['named_parameters']['quantizer.embedding.weight'],idx_neighbours)

        elif 'neighbour_weighing' in pred and pred['neighbour_weighing'] is not None: # For SOM with Gaussian kernel
            all_z_cont =  pred['all_z_cont']
            neighbour_weighing = pred['neighbour_weighing'] #[bs x sqrt(N) x sqrt(N)]
            bs = neighbour_weighing.shape[0]
            z_disc_neighbours = kwargs['named_parameters']['quantizer.embedding.weight'].repeat((bs,1,1)) #Use all nodes

        z_cont = all_z_cont[-1] if isinstance(all_z_cont,list) else all_z_cont
        z_disc_neighbours = z_disc_neighbours.permute(1,0,2) #from [bs x nr neighbours x F] to [nr neighbours x bs x F]
        return z_disc_neighbours, z_cont, neighbour_weighing

    def compute_loss(self,z_disc_neighbours, z_cont, neighbour_weighing):
        
        if neighbour_weighing is not None: #use the weighing matrix

            #neighbour_weighing [bs, sqrt(N), sqrt(N)]
            #z_disc_neighbours [neighbours,bs, F]
            #z_cont [bs,F]

            nr_neighbours = z_disc_neighbours.shape[0]

            if self.similarity_metric == 'mse':
                if self.detach_from_encoder:
                    SOM_loss = F.mse_loss(z_disc_neighbours,z_cont.unsqueeze(0).repeat((nr_neighbours,1,1)).detach(), reduction='none').permute(1,0,2) #[bs, neighbours, F]
                else:
                    SOM_loss = F.mse_loss(z_disc_neighbours,z_cont.unsqueeze(0).repeat((nr_neighbours,1,1)), reduction='none').permute(1,0,2) #[bs, neighbours, F]
            elif self.similarity_metric == 'mae':
                if self.detach_from_encoder:
                    SOM_loss = F.l1_loss(z_disc_neighbours,z_cont.unsqueeze(0).repeat((nr_neighbours,1,1)).detach(), reduction='none').permute(1,0,2) #[bs, neighbours, F]
                else:
                    SOM_loss = F.l1_loss(z_disc_neighbours,z_cont.unsqueeze(0).repeat((nr_neighbours,1,1)), reduction='none').permute(1,0,2) #[bs, neighbours, F]        
            else:
                raise ValueError('only mae and mse are implemented in combiation with the Gaussian kernel')

            weighing_grid_r = torch.transpose(neighbour_weighing,dim0=1,dim1=2).reshape((-1,nr_neighbours)) #[bs, neighbours]
            SOM_loss = torch.sum(weighing_grid_r.unsqueeze(-1) * SOM_loss, 1) #Sum the weighted neighbours #[bs, F]

            assert len(SOM_loss.shape) == 2
            SOM_loss = torch.mean(SOM_loss, dim=-1, keepdim=False)  #[bs]      

        else:    
            SOM_loss = 0 
            for z_disc in z_disc_neighbours: 
                if self.similarity_metric == 'mse':
                    if self.detach_from_encoder:
                        SOM_loss = SOM_loss + F.mse_loss(z_disc,z_cont.detach(), reduction='none') #[bs, features] 
                    else:
                        SOM_loss = SOM_loss + F.mse_loss(z_disc,z_cont, reduction='none') #[bs, features] 
                elif self.similarity_metric == 'mae':
                    if self.detach_from_encoder:
                        SOM_loss = SOM_loss + F.l1_loss(z_disc,z_cont.detach(), reduction='none') #[bs, features] 
                    else:
                        SOM_loss = SOM_loss + F.l1_loss (z_disc,z_cont, reduction='none') #[bs, features] 
                elif self.similarity_metric == 'dot':
                    if self.detach_from_encoder:
                        SOM_loss = SOM_loss + torch.matmul(z_cont.detach(),z_disc) #[bs]
                    else:
                        SOM_loss = SOM_loss + torch.matmul(z_cont,z_disc) #[bs]
                else:
                    raise ValueError('only mae, mse and dot product are implemented')

            if self.similarity_metric == 'mse':
                assert len(SOM_loss.shape) == 2
                SOM_loss = torch.mean(SOM_loss, dim=-1, keepdim=False)         #[bs]      
                
        return SOM_loss

    def forward(self, pred, **kwargs):
        """
        Args:
            pred (dict): Contains the output variables of the model. In case of an AR model, some of the output are a list, of which we only need to use the last entry/window.
        """
        z_disc_neighbours, z_cont, neighbour_weighing= self.prepare_all_preds(pred, **kwargs)
        SOM_loss = self.compute_loss(z_disc_neighbours, z_cont, neighbour_weighing)
        assert len(SOM_loss.shape) == 1
        if self.reduction == 'sum':
            SOM_loss = torch.sum(SOM_loss)
        else:
            raise NotImplementedError        

        return SOM_loss, {}


class temporal_SOM_loss(nn.Module):
    def __init__(self, reduction='sum',**kwargs):
        super().__init__()
        self.reduction = reduction
        self.softmax = torch.nn.Softmax(dim=-1)     
    
    def gather_nd(self, params, idx):
        idx = idx.long()
        outputs = []
        for i in range(len(idx)):
            outputs.append(params[[idx[i][j] for j in range(idx.shape[1])]])
        outputs = torch.stack(outputs)
        return outputs

    def select_2d_coordinate_from_tensor(self, tensor_input, coordinates, first_sel_dim, second_sel_dim):
        """[summary]

        Args:
            tensor_input (torch.Tensor): nD tensor 
            coordinates (torch.tensor): Tensor of coordinates of size 2xnr_coordinates
            first_sel_dim (int): Dimension related to the first coordinate
            second_sel_dim (int): Dimension related to the second coordinate.

        Returns:
            sel_vals (torch.Tensor): The vals from the input tensor, indicated by the given coordinates. 
        """
        sel_vals = []
        for coord in range(coordinates.shape[-1]):
            sel_vals.append(torch.index_select(torch.index_select(tensor_input, first_sel_dim,  coordinates[:,coord][0]), second_sel_dim , coordinates[:,coord][1]))
        return torch.cat(sel_vals)
        
    def interpret_pred(self, pred, **kwargs):
        all_sel_nodes, distance_matrix =  pred['all_codebook_idxs'], pred['distance_matrix']

        som_dim = math.sqrt(distance_matrix.shape[-1]) 

        # i) Get all old and new kx and ky's, it is important to split them out as kx and ky because of the neighbourhood structure. Using a one-D representation of k does not take this into account 
        all_k_x, all_k_y = [],[]
        if not isinstance(all_sel_nodes, list):
            all_sel_nodes = [all_sel_nodes]

        for k in all_sel_nodes:
            all_k_x.append(k // som_dim)
            all_k_y.append(k % som_dim)
        
        return all_sel_nodes, som_dim, all_k_x, all_k_y #, log_probs

class transition_loss(temporal_SOM_loss):
    def __init__(self, reduction='sum', **kwargs):
        super().__init__(reduction)

    def forward(self, pred, named_parameters, **kwargs):
        """
        Args:
            pred (dict):
        """
        # all_k contains a list of 1 or 2 vectors of size [bs] and denotes the selected nodes for each element in the batch.
        # In case of 2 lists, the first list denotes the selected nodes, one window earlier. 
        all_sel_nodes = pred['all_codebook_idxs']
        probs = self.softmax(named_parameters['quantizer.log_probs'])

        # Same implementations, but the second one is much faster.
        #sel_probs = self.select_2d_coordinate_from_tensor(probs, torch.stack([all_sel_nodes[0], all_sel_nodes[1]]), first_sel_dim=0, second_sel_dim=1).squeeze()
        sel_probs = probs[all_sel_nodes[0], all_sel_nodes[1]] 

        # The transition loss is -the expected value of the log of the probabilities 
        # To minimize this loss, we want to maximize the transition probabilities of the observed transitions
        transition_loss = -torch.log(sel_probs+1e-7) 
        
        if self.reduction == 'sum':
          return torch.sum(transition_loss,0),{}
        elif self.reduction == 'mean':
          raise ValueError


class smoothness_loss(temporal_SOM_loss):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, named_parameters, **kwargs):
        """
        Args:
            pred (dict): 
        """
        all_sel_nodes, distance_matrix = pred['all_codebook_idxs'], pred['distance_matrix']

        # Find the probabilities for each of the nodes to be selected at the current timestamp, given the previously selected node at t-1 (all_k[0]). 
        # Select the selected nodes of the penultimate window (i.e. -2), and slice on the zeroth axis from the log probs to acquire the conditional probabilities from these selected nodes to the next nodes.
        sel_probs = self.softmax(named_parameters['quantizer.log_probs'])[all_sel_nodes[-2]] 
       
        # distance_matrix: Provides the MSE between all codebook vectors and the last continuous embedding. Shape: [bs x som_nodes]
        smoothness_loss =  torch.sum(sel_probs*distance_matrix,-1)
        
        if self.reduction == 'sum':
          return torch.sum(smoothness_loss,0),{}
        elif self.reduction == 'mean':
          raise ValueError

