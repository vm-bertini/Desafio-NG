import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class SOM(nn.Module):
    def __init__(self, dim_1, dim_2, input_dim, circular=False):
        super(SOM, self).__init__()
        self.t_dim = dim_1 * dim_2
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.input_dim = input_dim
        self.circular = circular
        self.weights = nn.Parameter(torch.randn(self.t_dim, input_dim))
        self.locations = []
        self.location()
        # Set initial values for SOM training parameters
        self.gamma = max(1, self.t_dim // 2)  # Decay constant for learning rate and neighborhood
        self.sigma = max(1, self.t_dim // 2)  # Initial neighborhood radius
        
        if self.circular and not (self.dim_1 == 1 or self.dim_2 == 1):
            raise ValueError("SOM can only be circular if dim_1 == 1 or dim_2 == 1")
        if self.dim_1 < 1 or self.dim_2 < 1:
            raise ValueError("dim_1 and dim_2 must be >= 1")
        if self.input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        
    def location(self):
        # Create a list of (i, j) tuples representing the grid locations
        if self.dim_1 == 1:
            self.locations = [(0, j) for j in range(self.dim_2)]
        elif self.dim_2 == 1:
            self.locations = [(i, 0) for i in range(self.dim_1)]
        else:
            self.locations = [(i, j) for i in range(self.dim_1) for j in range(self.dim_2)]
            
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        distances = torch.cdist(x, self.weights)
        bmu_indices = torch.argmin(distances, dim=1)
        bmu_locations = [self.locations[i.item()] for i in bmu_indices]
        return bmu_locations
    def find_bmu(self, x):
        dists = torch.norm(self.weights - x, dim=1)
        return torch.argmin(dists).item()
    
    def update_weights(self, x, bmu_index, learning_rate, neighborhood_radius):
        bmu_location = self.locations[bmu_index]
        for i, loc in enumerate(self.locations):
            if self.circular:
                dist = min(abs(loc[0] - bmu_location[0]), self.dim_1 - abs(loc[0] - bmu_location[0])) + \
                       min(abs(loc[1] - bmu_location[1]), self.dim_2 - abs(loc[1] - bmu_location[1]))
            else:
                dist = np.linalg.norm(np.array(loc) - np.array(bmu_location))
                
            if dist <= neighborhood_radius:
                influence = np.exp(-dist**2 / (2 * (neighborhood_radius**2)))
                with torch.no_grad():
                    self.weights[i] = self.weights[i] + learning_rate * influence * (x - self.weights[i])
       
    def graph_circular_over_epochs(self, epoch, data):
        # Calculate the interval for plotting (every fifth of total epochs)
        interval = max(1, self.epochs // 5)
        is_first = epoch == 1
        is_last = epoch == self.epochs
        is_interval = (epoch % interval == 0)
        if is_first or is_last or is_interval:
            plt.figure(figsize=(8, 6))
            plt.title(f'SOM Circular Graph at Epoch {epoch}')
            weights_np = self.weights.detach().cpu().numpy()
            # Plot SOM nodes
            plt.scatter(weights_np[:, 0], weights_np[:, 1], color='red', label='SOM Nodes', marker='x')
            # Plot connections (circular)
            for i in range(len(weights_np)):
                next_i = (i + 1) % len(weights_np)
                plt.plot([weights_np[i, 0], weights_np[next_i, 0]], [weights_np[i, 1], weights_np[next_i, 1]], 'b--', alpha=0.5)
            # Plot cities
            data_np = np.array(data)
            plt.scatter(data_np[:, 0], data_np[:, 1], color='blue', label='Cities')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid()
            plt.show()
            
    def graph_epoch_print(self, epoch, data):
        print(f'Epoch {epoch}/{self.epochs} completed.')
        if self.circular:
            self.graph_circular_over_epochs(epoch, data)
            
    def graph_final_print(self,data):
        print('Final SOM weights:')
        print(self.weights.detach().cpu().numpy())
        if self.circular:
            plt.figure(figsize=(8, 6))
            plt.title('Final SOM Circular Graph')
            weights_np = self.weights.detach().cpu().numpy()
            # Find the N (number of cities) closest SOM nodes to the cities (forwarded nodes)
            weights_np = self.weights.detach().cpu().numpy()
            data_np = np.array(data)
            N = len(data_np)
            # Compute pairwise distances between SOM nodes and cities
            dists = np.linalg.norm(weights_np[:, None, :] - data_np[None, :, :], axis=2)
            # For each city, find the closest SOM node index
            closest_node_indices = np.argmin(dists, axis=0)
            # Get unique node indices (in case some nodes are closest to multiple cities)
            unique_node_indices = np.unique(closest_node_indices)
            # Plot only the forwarded nodes
            plt.scatter(weights_np[unique_node_indices, 0], weights_np[unique_node_indices, 1], color='red', label='Forwarded SOM Nodes', marker='x')
            # Plot connections (circular) among forwarded nodes
            for idx in range(len(unique_node_indices)):
                i = unique_node_indices[idx]
                next_i = unique_node_indices[(idx + 1) % len(unique_node_indices)]
                plt.plot([weights_np[i, 0], weights_np[next_i, 0]], [weights_np[i, 1], weights_np[next_i, 1]], 'b--', alpha=0.5)
            for i in range(len(weights_np)):
                next_i = (i + 1) % len(weights_np)
                plt.plot([weights_np[i, 0], weights_np[next_i, 0]], [weights_np[i, 1], weights_np[next_i, 1]], 'b--', alpha=0.5)
            # Plot cities
            data_np = np.array(data)
            plt.scatter(data_np[:, 0], data_np[:, 1], color='blue', label='Cities')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid()
            plt.show()
    def train_som(self, data, epochs):
        self.epochs = epochs
        for epoch in range(epochs):
            learning_rate = np.exp(-epoch / self.gamma)
            neighborhood_radius = self.sigma * np.exp(-epoch / self.gamma)
            for x in data:
                x = torch.tensor(x, dtype=torch.float32)
                bmu_index = self.find_bmu(x)
                
                # Update weights
                self.update_weights(x, bmu_index, learning_rate, neighborhood_radius)
            
            self.graph_epoch_print(epoch+1, data)
        self.graph_final_print(data)