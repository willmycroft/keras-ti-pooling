"""
    A wrapper for Keras networks, forcing them to act invariantly which
    respect to a transformation.
"""
from keras.layers import Input, maximum

def transformation_invariant(network, transformation, merge_type=maximum):
    """
        Wraps the network to be invariant with respect to the provided 
        transformation.
        
        # Arguments
          network: An uncompiled sequential Keras network.
          transformation: A subclass of Transformation.
          merge_type: A functional Keras merge layer used to combine the
                      output from the transformation networks. Should be
                      invariant under the action of the symmetric group.
                      For example, maximum, minimum, add, multiply, average, 
                      dot.
          
        # Returns
          list: A list of inputs for the transformation invariant network.
                Each input takes the data under a different transformation.
                The length of the list is equal to equal to 
                transformation.num_transformations(input_shape).
          tensor: The encoded output from the transformation invariant network.
                
    """
    input_shape = network.input_shape[1:]
    inputs = [Input(input_shape) for _ in range(transformation.num_transformations(input_shape))]
    encoded = [network(input_) for input_ in inputs]
    ti_tensor = merge_type(encoded)    
    return inputs, ti_tensor    