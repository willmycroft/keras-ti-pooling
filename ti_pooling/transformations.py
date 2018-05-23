"""
Transformations acting on data sets.
"""

import numpy as np
import itertools
import math

class Transformation():
    """
        The base class for all transformations to inherit from.
    """    
    
    def transform(self, data):
        """
            Transforms the data.
            
            # Arguments
              data: The data to be transformed, as a numpy array
              
            # Returns
              list: The transformed data. The length of the list should be
                    equal to self.num_transformations(data.shape).
                    
        """
        raise NotImplementedError()
        
    def num_transformations(self, input_shape):
        """
            Returns the total number of transformations.
            
            # Arguments
              input_shape: The shape of each data point to be transformed.
              
            # Returns
              int: The total number of transformations.
        """
        raise NotImplementedError()


class SymmetricGroup(Transformation):
    """
        Acts on a data set by permuting the elements of each vector. Only
        defined for two-dimensional data sets.
        
        Example: (1,2,3) -> [(1,2,3), (1,3,2), (2,1,3), 
                             (2,3,1), (3,1,2), (3,2,1)]
    """
    def transform(self, data):
        assert len(data.shape) == 2
        permute = np.array([list(itertools.permutations(x)) for x in data])
        return [permute[:,i] for i in range(permute.shape[1])]
    
    def num_transformations(self, input_shape):
        return math.factorial(input_shape[0])
    
    
class CyclicGroup(Transformation):
    """
        Acts on a data set by cycling the elements of each vector. Only
        defined for two-dimensional data sets.
        
        Example: (1,2,3) -> [(1,2,3), (3,1,2), (3,2,1)]
    """
    def transform(self, data):
        assert len(data.shape) == 2
        return [np.hstack([data[:, i:], data[:, :i]]) for i in range(data.shape[1])]
    
    def num_transformations(self, input_shape):
        return input_shape[0]
    
class D4(Transformation):
    """
        Acts on a data set via the action of a dihedral group with 8 elements.
        Only defined for data sets which are square in the axes we act upon.
        
        # Arguments
        axes: The axes of the data point we act on.
    """
    def __init__(self, axes=(0,1)):
        self.axes = (axes[0]+1, axes[1]+1) # For acting on entire dataset.
    
    def transform(self, data):
        assert data.shape[self.axes[0]] == data.shape[self.axes[1]]
        rotated = [np.rot90(data, r, axes=self.axes) for r in range(4)]
        return rotated + [np.flip(x, axis=self.axes[0]) for x in rotated]
    
    def num_transformations(self, input_shape):
        return 8