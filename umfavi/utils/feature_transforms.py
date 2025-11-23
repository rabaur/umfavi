import numpy as np
from typing import Callable


def to_one_hot(discr_x: int, n: int) -> np.ndarray:
    one_hot = np.zeros(n)
    one_hot[discr_x] = 1
    return one_hot

def apply_transform(transform: Callable, x: np.ndarray) -> np.ndarray:
    """
    Apply a transform function to each element of an array.
    
    This function handles transforms that return arrays (e.g., one-hot encoding)
    where np.vectorize would fail with "setting an array element with a sequence".
    
    Args:
        transform: Function to apply to each element. Can return scalars or arrays.
        x: Input array of shape (..., feature_dim)
        
    Returns:
        Transformed array of shape (..., new_feature_dim)
        where new_feature_dim depends on the transform output.
        
    Example:
        >>> x = np.array([[[0], [1]], [[1], [0]]])  # shape: (2, 2, 1)
        >>> transform = lambda a: to_one_hot(int(a), 2)
        >>> result = apply_transform(transform, x)
        >>> result.shape
        (2, 2, 2)  # Last dimension expanded from 1 to 2
    """
    original_shape = x.shape
    
    # Flatten to 1D for easy iteration
    x_flat = x.reshape(-1)
    
    # Apply transform to each element
    transformed_list = [transform(int(elem)) for elem in x_flat]
    
    # Check if transform returns arrays or scalars
    if len(transformed_list) > 0:
        first_result = transformed_list[0]
        
        # Determine the feature dimension of the output
        if isinstance(first_result, (np.ndarray, list, tuple)):
            new_feat_dim = len(first_result)
        else:
            new_feat_dim = 1
        
        # Stack all transformed results
        transformed_array = np.array(transformed_list)
        
        # Reshape back to original structure with new feature dimension
        # Original shape: (..., old_feat_dim) -> New shape: (..., new_feat_dim)
        new_shape = original_shape[:-1] + (new_feat_dim,)
        transformed_array = transformed_array.reshape(new_shape)
        
        return transformed_array
    else:
        # Empty array case
        return x
