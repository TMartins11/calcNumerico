"""
main.py
Plano de Investigação – Sistemas Lineares
Disciplina: Cálculo Numérico
Alunos: Thiago Martins da Silva e Pedro Augusto de Souza Finochio
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from gauss import resolver_gauss, gauss, resolver_sem_pivo

# ==============================================================================
# 1. Eliminação de Gauss e Pivoteamento

# ==============================================================================
# 1.1 Verificação básica
# Q1.1. — Sistema: 3x1 + 2x2 + 4x3 = 1, x1 + x2 + 2x3 = 2, 4x1 + 3x2 − 2x3 = 3.
# ==============================================================================

A = np.array([
    [3, 2, 4],
    [1, 1, 2],
    [4, 3, -2]
], dtype=float)

b = np.array([1, 2, 3], dtype=float)

x = resolver_gauss(A, b)

print(f"Q1.1. — Solução: {x}")
print(f"Q1.1. — Resíduo: {np.linalg.norm(A @ x - b):.2e}")

# ==============================================================================
# Q1.2. — Matriz U
# ==============================================================================
U, _ = gauss(A.copy(), b.copy())

print("\nQ1.2. — Matriz U:")
print(U)

print("\nObservação:")
print("A linha 1 foi trocada com a linha 3 (pivoteamento parcial).")


# ==============================================================================
# 1.2 Efeito do pivoteamento na precisão
# Q1.3. — Versão sem pivoteamento e com pivoteamento parcial ao sistema: 
#  [[0.0003, 3],[1, 1]] * [[x1, x2]] = [[2.0001, 1]] 
# aritmética float32 (dtype=np.float32) para simular precisão reduzida. 
# ==============================================================================

# Sistema mal-condicionado (float32)
A2 = np.array([
    [0.0003, 3],
    [1, 1]
], dtype=np.float32)

b2 = np.array([2.0001, 1], dtype=np.float32)

# Soluções
x_sem = resolver_sem_pivo(A2, b2)
x_com = resolver_gauss(A2, b2)

# Solução exata
x_exato = np.array([1/3, 2/3], dtype=np.float32)

# Erros relativos
erro_sem = np.abs((x_sem - x_exato) / x_exato)
erro_com = np.abs((x_com - x_exato) / x_exato)

print("\nQ1.3. — Efeito do pivoteamento")

print("\nSolução exata:", x_exato)

print("\n--- Sem pivoteamento ---")
print("Solução:", x_sem)
print(f"Erro relativo x1: {erro_sem[0]:.2e}")
print(f"Erro relativo x2: {erro_sem[1]:.2e}")

print("\n--- Com pivoteamento ---")
print("Solução:", x_com)
print(f"Erro relativo x1: {erro_com[0]:.2e}")
print(f"Erro relativo x2: {erro_com[1]:.2e}")

# ==============================================================================
# Q1.4 — Erro vs a11 (escala log-log)
# ==============================================================================

a11_vals = np.array([1e-1, 1e-3, 1e-6, 1e-9])

erro_sem_lista = []
erro_com_lista = []

for a11 in a11_vals:
    A_temp = np.array([
        [a11, 3],
        [1, 1]
    ], dtype=np.float32)

    b_temp = np.array([2.0001, 1], dtype=np.float32)

    x_sem = resolver_sem_pivo(A_temp, b_temp)
    x_com = resolver_gauss(A_temp, b_temp)

    x_exato = np.array([1/3, 2/3], dtype=np.float32)

    erro_sem = abs((x_sem[0] - x_exato[0]) / x_exato[0])
    erro_com = abs((x_com[0] - x_exato[0]) / x_exato[0])

    erro_sem_lista.append(erro_sem)
    erro_com_lista.append(erro_com)

idx = np.argsort(a11_vals)

a11_vals = a11_vals[idx]
erro_sem_lista = np.array(erro_sem_lista)[idx]
erro_com_lista = np.array(erro_com_lista)[idx]

plt.figure()

plt.loglog(a11_vals, erro_sem_lista, 'o-', label='Sem pivoteamento')
plt.loglog(a11_vals, erro_com_lista, 's-', label='Com pivoteamento')

plt.xlabel('a11')
plt.ylabel('Erro relativo em x1')
plt.title('Erro relativo de x1 em função de a11')

plt.legend(frameon=False)

plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("erro_relativo_a11.pdf", dpi=150)

# ==============================================================================
# 1.3 Sistemas Singulares e quase-singulares
# Q1.5 — Aplique resolver_gauss ao Sistema Singular: 
# x − 3y + z = 1, 6x − 18y + 4z = 2, −x + 3y − z = 4
# Limiar: |a_kk| < 10^{-12} 
# ==============================================================================

print("\nQ1.5. — Sistema singular")

A_sing = np.array([
    [1, -3, 1],
    [6, -18, 4],
    [-1, 3, -1]
], dtype=float)

b_sing = np.array([1, 2, 4], dtype=float)

try:
    x_sing = resolver_gauss(A_sing, b_sing)
    print("Solução:", x_sing)
except ValueError as e:
    print("Erro detectado:", e)

# ==============================================================================

# TODO: 2. Fatoração LU (Doolittle)
# TODO: 3. Fatoração de Cholesky para Matrizes SPD
# TODO: 4. Algoritmo de Thomas para Sistemas Tridiagonais
# TODO: 5. Custo Computacional Empírico
# TODO: 6. Condicionamento e Sensibilidade à Perturbação
# TODO: 7. Projeto Integrador: PageRank Numérico
# TODO: 8. Desafio (Opcional — Pontuação Extra)

plt.show()