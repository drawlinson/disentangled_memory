import torch

def nonzero_bit_sequences(N, device=None):
    # integers 1 .. 2^N - 1
    ints = torch.arange(1, 2**N, device=device)
    
    # convert to bits
    bits = ((ints[:, None] >> torch.arange(N, device=device)) & 1).to(torch.long)
    
    return bits

def one_hot_bit_sequences(N, device=None):
    return torch.eye(N, device=device)