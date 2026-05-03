import numpy as np

def gauss(A, b):
    """
    Eliminação de Gauss com pivoteamento parcial.
    Retorna (A_triangular, b_modificado).
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    for k in range(n - 1):
        # pivoteamento parcial: busca maior |a_ik| abaixo da linha k
        p = np.argmax(np.abs(A[k:, k])) + k
        
        # troca de linhas
        A[[k, p]] = A[[p, k]]
        b[[k, p]] = b[[p, k]]

        # detecção de singularidade
        if abs(A[k, k]) < 1e-12:
            raise ValueError("Matriz singular ou quase singular (pivô ~ 0)")

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]  # multiplicador
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    # checagem do último pivô
    if abs(A[n-1, n-1]) < 1e-12:
        raise ValueError("Matriz singular ou quase singular (pivô final ~ 0)")

    return A, b


def subst_retro(A, b):
    """
    Substituição retroativa para sistema triangular superior.
    """
    n = len(b)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i+1:] @ x[i+1:]) / A[i, i]

    return x


def resolver_gauss(A, b):
    """
    Pipeline completo: Gauss + substituição retroativa.
    """
    Au, bu = gauss(A, b)
    return subst_retro(Au, bu)


def gauss_sem_pivo(A, b):
    A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    n = len(b)

    for k in range(n - 1):
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    return A, b


def resolver_sem_pivo(A, b):
    U, b_mod = gauss_sem_pivo(A, b)
    return subst_retro(U, b_mod)