from torch import nn
from torchviz import make_dot

def write_loss_plot(trainable_modules, loss, filename:str = "graph"):
    params = dict(trainable_modules.named_parameters())
    g = make_dot(loss, params=params)         
    g.render(filename, format="png") 

def create_functional_nonlinearity(nonlinearity:str):
    """
    Returns the functional form of a torch nonlinearity. 
    Just allows interpretation of this config option to be centralized.
    """
    if nonlinearity is None:
        return None
    if nonlinearity == "none":
        return None  
    if nonlinearity == "relu":
        return nn.ReLU()
    if nonlinearity == "leaky-relu":
        return nn.LeakyReLU()
    if nonlinearity == "gelu":
        return nn.GELU()
    raise ValueError("Nonlinearity not recognized.")

