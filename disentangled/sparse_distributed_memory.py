import torch
import torch.nn as nn
import math
    

class SparseDistributedMemory(nn.Module):
    """
    A fast-learning vector key --> vector value memory. 
    The representation is sparse and distributed for combinatoric capacity and robust
    value retention.    

    Trained by direct gradient descent (not SGD), additive updates, minibatch-compatible 
    but not required.

    TODO: Increase capacity by recruiting idle "cells":
    - Very slowly, add weight to never-used cells.
    - Possibly remove weight from "synapses" which are never used when cells are active..
    """

    def __init__(
        self,
        input_size: int,
        memory_size: int = 65536,
        sparsity: int = 32,
        value_size: int = 1,
        learning_rate:float = 0.1,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        self.input_size = input_size
        self.memory_size = memory_size
        self.sparsity = sparsity
        self.learning_rate = learning_rate

        # Fixed random projection (not trained)
        projection = torch.randn(memory_size, input_size, device=device, dtype=dtype)
        projection = projection / projection.norm(dim=1, keepdim=True)
        self.register_buffer("proj", projection)

        # Stored value vectors
        self.register_buffer("mem_value", torch.zeros((memory_size, value_size), device=device, dtype=dtype))

    def reset_values(self):
        self.mem_value.fill_(0)

    def encode(self, keys: torch.Tensor):
        """
        keys: [B, input_size]
        returns indices: [B, sparsity]
        """
        # Project
        scores = keys @ self.proj.T  # [B, memory_size]

        # Top-k sparse code
        _, indices = torch.topk(scores, self.sparsity, dim=1)
        return indices

    def get_value(self, keys: torch.Tensor):
        """
        keys: [B, input_size]
        returns values: [B]
        """
        keys = keys.detach()
        indices = self.encode(keys)  # [B, sparsity]
        return self.get_value_with_indices(indices)
    
    def get_value_with_indices(self, indices: torch.Tensor):
        v_out = self.mem_value[indices].sum(dim=1)
        return v_out

    def set_value(self, keys: torch.Tensor, targets: torch.Tensor):
        """
        keys:    [B, input_size]
        targets: [B]   (scalar per key)
        """
        keys = keys.detach()
        indices = self.encode(keys)  # [B, sparsity]
        self.set_value_with_indices(indices, targets)

    def set_value_with_indices(self, indices: torch.Tensor, targets: torch.Tensor):
        """
        keys:    [B, input_size]
        targets: [B, value_size]
        """
        B, sparsity = indices.shape
        V = targets.shape[1]
        # /sparsity because the retrieval target is the sum, so individually should be 1/k of that.
        retrieved = self.mem_value[indices].sum(dim=1)        # [B, V]
        deltas = ((targets - retrieved) / sparsity) * self.learning_rate  # [B, V]

        expanded_delta = deltas.unsqueeze(1).expand(B, sparsity, V).reshape(-1, V)
        expanded_indices = indices.reshape(-1,1).expand(-1, V)
        self.mem_value.scatter_add_(0, expanded_indices, expanded_delta)


class AssociativeSparseDistributedMemory(nn.Module):
    """
    Uses one SparseDistributedMemory to remember the active cliques of another.
    This allows clique memorization, clique cueing and completion, and sampling from the 
    combined memory.
    """

    def __init__(
        self,
        input_size: int,
        value_size: int = 1,
        value_capacity: int = 2000,
        value_sparsity: int = 32,
        associative_capacity: int = 2000,
        associative_sparsity: int = 32,
        associative_projection_scale_factor:int = 1,  # 2, 4, etc
        learning_rate:float = 0.1,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()

        self.value_memory = SparseDistributedMemory(
            input_size=input_size,  # size of input "key"
            memory_size=value_capacity,  # number of "slots"
            sparsity=value_sparsity,  # number of active bits in encoded key
            value_size=value_size,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )

         # Projecting the identities of the clique allows the pattern to be stored
         # in a distributed way.
        projected_size = associative_capacity * associative_projection_scale_factor
        self._create_clique_encoder(
            input_size = value_capacity, 
            projected_size = projected_size,
            device=device,
        )

        self.associative_memory = SparseDistributedMemory(
            input_size=projected_size,  # size of input "key"
            memory_size=associative_capacity,  # number of "slots"
            sparsity=associative_sparsity,  # number of active bits in encoded key
            value_size=value_capacity,
            learning_rate=learning_rate,
            device=device,
            dtype=dtype,
        )

    def _create_input_encoder(
        self, 
        index:int,
        input_size:int,
        output_size:int,
        device,
        dtype=torch.float32,
    ):
        encoder = torch.randn(output_size, input_size, device=device, dtype=dtype)
        encoder = encoder / encoder.norm(dim=1, keepdim=True)
        buffer_name = self._get_partition_name(index)
        self.register_buffer(buffer_name, encoder)        

    def _get_partition_name(self, partition_index:int) -> str:
        buffer_name = "index_projection" + str(partition_index)
        return buffer_name
    
    def _create_clique_encoder(
            self, 
            input_size:int, 
            projected_size:int,
            device,
            dtype=torch.float32,
    ):
        emb = torch.randn(input_size, projected_size, device=device, dtype=dtype)
        emb /= math.sqrt(projected_size)
        self.register_buffer("clique_encoder", emb)        

    def encode(self, keys: torch.Tensor):
        return self.value_memory.encode(keys)

    def get_value(self, keys: torch.Tensor):
        """
        keys: [B, input_size]
        returns values: [B]
        """
        clique_indices = self.encode(keys)
        return self.value_memory.get_value_with_indices(clique_indices)

    def set_value(self, keys: torch.Tensor, targets: torch.Tensor):
        clique_indices = self.encode(keys)  # [B, sparsity]
        self.value_memory.set_value_with_indices(clique_indices, targets)

        # Want a *distributed* encoding of the indices as values.
        # To make it distributed, must shuffle/randomize the indices.
        # p must be dense, gets encoded again
        clique_projection = self._dense_projection(clique_indices)
        clique_vectors = self._get_clique_vectors(clique_indices)
        self.associative_memory.set_value(
            keys=clique_projection,
            targets=clique_vectors,  # store exactly as the pattern to memorize.
        )

    def _get_clique_vectors(self, clique_indices):
        B = clique_indices.shape[0]
        V = self.associative_memory.mem_value.shape[1]
        clique_vectors = torch.zeros(
            B, 
            V,#self.num_indices,
            device=self.clique_encoder.device,
        ).scatter_(
            dim=1, # scatter into dim=1
            index=clique_indices, 
            value=1.0,  # values
        )  # potentially reuse buffer with zero_
        return clique_vectors
    
    def get_cliques(self, keys: torch.Tensor):
        clique_indices = self.encode(keys)  # [B, sparsity]
        clique_projection = self._dense_projection(clique_indices)
        cliques = self.associative_memory.get_value(
            keys=clique_projection,  
        )
        return cliques        

    def get_clique_indices(self, keys: torch.Tensor, t = 0.01):
        # TODO topk filter, since sparsity is known
        # TODO add additive bias against re-selecting same clique
        cliques = self.get_cliques(keys)
        indices = torch.nonzero(cliques > t)
        return indices

    def get_clique_indices_with_indices(self, clique_indices: torch.Tensor, t = 0.01):
        clique_projection = self._dense_projection(clique_indices)
        cliques = self.associative_memory.get_value(
            keys=clique_projection,  
        )
        indices = torch.nonzero(cliques > t)
        return indices

    def _dense_projection(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, K]  (sparse indices)
        Returns:
            dense: [B, proj_dim]
        """
        # [B, K, D]
        p = self.clique_encoder[indices]

        # Sum over active indices
        p = p.sum(dim=1)

        # Normalize by sqrt(K) to make scale invariant
        K = indices.size(1)
        p = p / math.sqrt(K)

        # Additional L2 norm (root sum sq)
        p = torch.nn.functional.normalize(p, dim=-1)
        return p