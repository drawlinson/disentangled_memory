from abc import ABC, abstractmethod

class TokenEnvironment(ABC):
    """
    An Environment which emits tokens as its observation.
    This interface is needed for Agent interactions.
    """

    @abstractmethod
    def get_token_key(self) -> str:
        """
        Return the key for the observation dict to get the token index observation values.
        """
        pass

    @abstractmethod
    def get_num_tokens(self) -> int:
        """
        Returns the number of unique tokens which can be observed.
        """
        pass
    
    @abstractmethod
    def get_tokens(self) -> list[str]:
        """
        Gets a list of all possible tokens with persistent ordering so that indices
        can be used in the token observation.        
        """
        pass
