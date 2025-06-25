from mcp.server.fastmcp import FastMCP
import torch
from SOM.SOM import SOM   
import os
from Functions.Read_TSP import read_tsp
import numpy as np
import base64
import io
import matplotlib.pyplot as plt

class MCPServer(FastMCP):
    def __init__(self, name = "MCP Server"):
        super().__init__()
        self._server_name = name

    def get_server_name(self):
        return self._server_name
    
#Instantiating the MCPServer class:
server = MCPServer(name="Servidor stdio MCP")
    
@server.tool()
async def Process_Map(map:list, circular: bool = True):
    """ Process a map using Self-Organizing Map (SOM) algorithm.
    Args:
        map (list): Input map as a serialized tensor (list of lists).
        circular (bool): Whether the SOM is circular (used for tsp solutions). Defaults to True.
    Returns:
        List (list): A tensor, list of lists containing the final positions from the TSP file."""

    map = torch.tensor(map, dtype=torch.float32)  # Ensure the input is a tensor
    #Checking if the map is empty:
    if map.numel() == 0:
        return "Map is empty. Please provide a valid map."
    #Checking if the map is a tensor:
    if not isinstance(map, torch.Tensor):
        return "Map must be a tensor. Please provide a valid tensor."
    
    #extracting dimensions from the map:
    input_dim = map.shape[1]
    dim_1 = map.shape[0]*3
    dim_2 = 1
    #Creating SOM instance:
    som = SOM(dim_1, dim_2, input_dim, circular)

    #Training SOM:
    som.train_som(map, epochs=300)
    #Getting the final weights:
    final_weights = som.forward(map)
    #Returning the final weights:
    return {"final_weights": final_weights.tolist()} # Convert to list for JSON serialization

@server.tool()
def Load_TSP_File(file_name: str = "", file_id: int = None):
    """Load a TSP file from the Local_TSP folder and create a tensor with positions.
    Args:
        file_name (str): The name of the TSP file to load. Takes priority if provided.
        file_id (int): The index of the TSP file in the folder, used if file_name is not provided. 
        If no tsp is offered set this value to random.
    Returns:
        List (list): A tensor, list of lists containing the positions from the TSP file.
    """
    tsp_folder = "./Local_TSP"

    # If file_name is empty, try to get file by file_id
    if not file_name:
        if file_id is None:
            return {"error": "No file_name or file_id provided."}
        # List all .tsp files in the folder
        files = [f for f in os.listdir(tsp_folder) if f.lower().endswith('.tsp')]
        if not files:
            return {"error": "No TSP files found in the folder."}
        if file_id < 0 or file_id >= len(files):
            return {"error": f"file_id {file_id} is out of range. Found {len(files)} files."}
        file_name = files[file_id]

    file_path = os.path.join(tsp_folder, file_name)

    # Load the TSP file and convert it to a tensor
    tensor = read_tsp(file_path)

    # Serialize the tensor to a list for JSON compatibility
    return {"tensor": tensor.tolist(), "Name": file_name}

@server.tool()
def gen_cities(n_cities:int):
    """Generate a list of cities with random coordinates.
    Args:
        n_cities (int): Number of cities to generate.
    Returns:
        obj (object): Contains a list of tuples with city coordinates."""
    cities = []
    for i in range(n_cities):
        rand = np.random.normal(0,1)
        cities.append((np.random.uniform(-1, 1)+rand*0.1, np.random.uniform(-1, 1)+rand*0.1))
    return {"Cities": cities}


@server.tool()
def graph_cities(cities:list, file_name: str = "cities_graph.png", circular: bool = False):
    """Graph a single cities list
    circular should be set to true if the cities data is part of the solution"""
    folder = "./PNGs/"
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    x, y = zip(*cities)
    plt.scatter(x, y)
    plt.title('Cities')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()

    if circular:
        plt.plot(x + (x[0],), y + (y[0],), linestyle='-', color='orange' if circular else 'blue')
    # Check if a file with the same name already exists and delete it
    if os.path.exists(folder+file_name):
        os.remove(folder+file_name)
    plt.savefig(folder+file_name)
    plt.close()
    return "Graph saved as"+ file_name

@server.tool()
def graph_mult_cities(cities_obj: object, file_name: str = "cities_graph.png", circular: list[bool] = None):
    """Graph multiple cities lists.
    
    The first list value is the cities and the second one is the SOM weights.
    'circular' should be a list of booleans, one for each cities set.
    """
    folder = "./PNGs/"
    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    if circular is None:
        circular = [False] * len(cities_obj)
    for i, cities in enumerate(cities_obj.values()):
        x, y = zip(*cities)
        plt.scatter(x, y)
        plt.title(f'Cities {i+1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid()
        if circular[i]:
            plt.plot(x + (x[0],), y + (y[0],), linestyle='-', color='red')
    # Check if a file with the same name already exists and delete it
    if os.path.exists(folder + file_name):
        os.remove(folder + file_name)
    plt.savefig(folder + file_name)
    plt.close()
    return "Graph saved as " + file_name
