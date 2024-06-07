import numpy as np

def calculate_A_SFAS(tensor1, tensor2):
    """
    Computes the Average Semantic Feature Alignment scores between two n-dimensional tensors.

    Parameters:
    tensor1 (numpy.ndarray): The first tensor.
    tensor2 (numpy.ndarray): The second tensor.

    Returns:
    float: The Average Semantic Feature Alignment scores between tensor1 and tensor2.
    """
    # Flatten the tensors to 1-dimensional arrays
    flat_tensor1 = tensor1.flatten()
    flat_tensor2 = tensor2.flatten()
    
    # Compute the dot product
    dot_product = np.dot(flat_tensor1, flat_tensor2)
    
    # Compute the magnitudes (norms) of the tensors
    norm_tensor1 = np.linalg.norm(flat_tensor1)
    norm_tensor2 = np.linalg.norm(flat_tensor2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_tensor1 * norm_tensor2)
    
    return cosine_sim

# Example usage:
tensor1 = np.array([[1, 2, 3], [4, 5, 6]])
tensor2 = np.array([[1, 0, 0], [0, 1, 0]])
similarity = cosine_similarity(tensor1, tensor2)
print("Cosine Similarity:", similarity)
