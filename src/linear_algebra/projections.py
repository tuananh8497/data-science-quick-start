import numpy as np

def scalar_projection(u, v):
    """
    Calculates the scalar projection of vector u onto vector v.

    Args:
        u (numpy.ndarray): The vector to be projected.
        v (numpy.ndarray): The vector onto which u is projected.

    Returns:
        float: The scalar projection of u onto v.
    """
    return (np.dot(u, v) / np.linalg.norm(v))

def vector_projection(u, v):
    """
    Calculates the vector projection of vector u onto vector v.

    Args:
        u (numpy.ndarray): The vector to be projected.
        v (numpy.ndarray): The vector onto which u is projected.

    Returns:
        numpy.ndarray: The vector projection of u onto v.
    """
    return (np.dot(u, v) / np.linalg.norm(v)**2) * v

if __name__ == '__main__':
    # Example usage:
    u = np.array([4, 2])
    v = np.array([1, 1])

    scalar_proj = scalar_projection(u, v)
    vector_proj = vector_projection(u, v)

    print("Scalar projection:", scalar_proj)
    print("Vector projection:", vector_proj)