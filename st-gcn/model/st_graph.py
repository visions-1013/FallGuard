import numpy as np

class ST_Graph:
    pass

def symnorm(A):
    """Compute a symmetrically normalized 
    adjacency matrix

    Args:
        A : adjacency matrix

    Returns:
        A_symnorm: symnormed A
    """
    D = np.diag(np.sum(A, axis=0))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    return D_inv_sqrt @ A @ D_inv_sqrt

def get_adjacency(edges, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A

def get_distance_adjacency(edges, num_node):
    I = np.identity(num_node)
    N = get_adjacency(edges, num_node)
    A = np.stack([I, symnorm(N)])
    return A

def get_uniform_adjacency(edges, num_node):
    I = np.identity(num_node)
    N = get_adjacency(edges, num_node)
    A = symnorm(I + N)
    return A[np.newaxis, :, :]

if __name__ == '__main__':
    # Test input
    num_nodes = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 1)]

    A = get_adjacency(edges, num_nodes)
    print(A)
    print(symnorm(A))
    print(np.sum(symnorm(A), axis=1))
    
    print(get_distance_adjacency(edges, num_nodes))