from mcp.server.fastmcp import FastMCP
import torch
from SOM.SOM import SOM   

class MCPServer(FastMCP):
    def __init__(self, name = "MCP Server"):
        super().__init__()
        self._server_name = name

    def get_server_name(self):
        return self._server_name
    
#Instantiating the MCPServer class:
server = MCPServer(name="Servidor stdio MCP")
    
@server.tool()
async def Process_Map(map, circular: bool = True):
    """ Process a map using Self-Organizing Map (SOM) algorithm.
    Args:
        map (torch.Tensor): Input map as a tensor.
        circular (bool): Whether the SOM is circular (used for tsp solutions). Defaults to True."""

    #Checking if the map is empty:
    if not map:
        return "Map is empty. Please provide a valid map."
    #Checking if the map is a tensor:
    if not isinstance(map, torch.Tensor):
        return "Map must be a tensor. Please provide a valid tensor."
    
    #extracting dimensions from the map:
    input_dim = map.shape[1]
    dim_1 = map.shape[0]*3
    dim_2 = 1
    #Creating SOM instance:
    som = SOM(dim_1, dim_2, circular, input_dim=input_dim)

    #Training SOM:
    som.train_som(map, epochs=300)
    #Getting the final weights:
    final_weights = som.forward(map)
    #Returning the final weights:
    return final_weights.tolist()  # Convert to list for JSON serialization