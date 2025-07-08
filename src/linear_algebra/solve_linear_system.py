import numpy as np

def solve_linear_system(A, b):
    """
    Solves a system of linear equations Ax = b using NumPy.

    Args:
        A (numpy.ndarray): The coefficient matrix.
        b (numpy.ndarray): The right-hand side vector.

    Returns:
        numpy.ndarray: The solution vector x, or None if the system has no unique solution.
    """
    try:
        x = np.linalg.solve(A, b)
        return x
    except np.linalg.LinAlgError:
        print("The system has no unique solution.")
        return None

if __name__ == '__main__':
    # Example usage:
    A = np.array([[2, 1], [1, 3]])
    b = np.array([5, 8])

    x = solve_linear_system(A, b)

    if x is not None:
        print("Solution:", x)