"""
main.py
Plano de Investigação – Sistemas Lineares
Disciplina: Cálculo Numérico
Alunos: Thiago Martins da Silva e Pedro Augusto de Souza Finochio
"""

import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
from scipy.linalg import lu as scipy_lu

from gauss import resolver_gauss, gauss, resolver_sem_pivo
from lu import fatoracao_lu, subst_prog, subst_retro
from thomas import thomas, montar_tridiagonal

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
# 2. Fatoração LU (Doolittle)

# ==============================================================================
# Q2.1. — Construção e verificação da fatoração LU para a matriz A
# ==============================================================================

print("\n" + "=" * 70)
print("Q2.1. — Fatoração LU: construção e verificação")
print("=" * 70)

A_lu = np.array([
    [2,  1, 1],
    [4, -6, 0],
    [-2, 7, 2]
], dtype=float)

L, U = fatoracao_lu(A_lu)

print("\nL =")
print(np.round(L, 6))

print("\nU =")
print(np.round(U, 6))

erro_fat = np.linalg.norm(L @ U - A_lu, 'fro')
print(f"\nErro de fatoração ||LU - A||_F = {erro_fat:.2e}")

# ==============================================================================
# Q2.2. — Resolver dois lados direitos reutilizando a mesma fatoração LU
# ==============================================================================

print("\n" + "=" * 70)
print("Q2.2. — Resolução com dois lados direitos (fatoração única)")
print("=" * 70)

b1 = np.array([1, 2, 3], dtype=float)
b2 = np.array([0, 1, -1], dtype=float)

# Fatora A uma única vez
L, U = fatoracao_lu(A_lu)

# Resolve para b1
y1 = subst_prog(L, b1)
x1 = subst_retro(U, y1)

# Reutiliza L e U para b2 (sem refatorar)
y2 = subst_prog(L, b2)
x2 = subst_retro(U, y2)

print(f"\nSolução x1 (b1 = {b1}): {x1}")
print(f"Resíduo ||A x1 - b1||_2 = {np.linalg.norm(A_lu @ x1 - b1):.2e}")

print(f"\nSolução x2 (b2 = {b2}): {x2}")
print(f"Resíduo ||A x2 - b2||_2 = {np.linalg.norm(A_lu @ x2 - b2):.2e}")

# ==============================================================================
# Q2.3. — Benchmark: Gauss repetido vs LU reutilizado (n=200, k=50)
# ==============================================================================

print("\n" + "=" * 70)
print("Q2.3. — Benchmark: Gauss repetido vs LU reutilizado")
print("=" * 70)

np.random.seed(42)
n_bench = 200
k_bench = 50

# Matriz aleatória não-singular (dominância diagonal garante inversibilidade)
A_bench = np.random.randn(n_bench, n_bench)
A_bench += n_bench * np.eye(n_bench)

bs_bench = [np.random.randn(n_bench) for _ in range(k_bench)]

# Estratégia (a): Gauss repetido — refatora A a cada sistema
t0 = time.perf_counter()
for b_k in bs_bench:
    resolver_gauss(A_bench, b_k)
t_gauss = time.perf_counter() - t0

# Estratégia (b): LU reutilizado — fatora A uma vez e só faz substituições
t0 = time.perf_counter()
L_b, U_b = fatoracao_lu(A_bench)
for b_k in bs_bench:
    y_k = subst_prog(L_b, b_k)
    subst_retro(U_b, y_k)
t_lu = time.perf_counter() - t0

fator_aceleracao = t_gauss / t_lu

print(f"\n{'Estratégia':<20} {'Tempo total (s)':>16} {'Tempo/sistema (ms)':>20}")
print("-" * 58)
print(f"{'Gauss repetido':<20} {t_gauss:>16.4f} {t_gauss / k_bench * 1000:>20.2f}")
print(f"{'LU reutilizado':<20} {t_lu:>16.4f} {t_lu  / k_bench * 1000:>20.2f}")
print(f"\nFator de aceleração: {fator_aceleracao:.1f}x")

# ==============================================================================
# Q2.4. — Fatoração PLU com pivoteamento via scipy para matriz com pivô nulo
# ==============================================================================

print("\n" + "=" * 70)
print("Q2.4. — Fatoração PLU (pivoteamento via scipy)")
print("=" * 70)

A_plu = np.array([
    [0, 1],
    [2, 3]
], dtype=float)

# (a) LU sem pivoteamento — pivô nulo causa divisão por zero silenciosa no
#     NumPy: não levanta exceção, mas produz nan/inf na matriz resultante.
#     Por isso suprimimos o RuntimeWarning e verificamos o resultado com
#     np.isfinite, que detecta nan/inf de forma explícita e controlada.
print("\n(a) Tentativa com fatoracao_lu (sem pivoteamento):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    L_plu, U_plu = fatoracao_lu(A_plu)

if not (np.isfinite(L_plu).all() and np.isfinite(U_plu).all()):
    print("Falha detectada: pivô nulo em U[0,0] = 0 causou divisão por zero.")
    print("  L contém nan/inf:", not np.isfinite(L_plu).all())
    print("  U contém nan/inf:", not np.isfinite(U_plu).all())
    print("  U[0,0] (pivô problemático):", U_plu[0, 0])
    print("Conclusão: sem pivoteamento, a fatoração LU falha silenciosamente")
    print("           quando o pivô diagonal é zero — o resultado é inválido.")
else:
    print("L =", L_plu)
    print("U =", U_plu)

# (b) scipy.linalg.lu — usa pivoteamento e retorna P tal que PA = LU
print("\n(b) Fatoração PLU via scipy.linalg.lu:")
P, L_plu, U_plu = scipy_lu(A_plu)

print("P =\n", P)
print("L =\n", L_plu)
print("U =\n", U_plu)
print(f"\nVerificação PA = LU: {np.allclose(P @ A_plu, L_plu @ U_plu)}")

# Resolução de Ax = b usando PLU: PA = LU => LUx = Pb
b_plu = np.array([1.0, 2.0])

Pb    = P @ b_plu                  # aplica a mesma permutação ao lado direito
y_plu = subst_prog(L_plu, Pb)     # Ly = Pb
x_plu = subst_retro(U_plu, y_plu) # Ux = y

print(f"\nResolução de Ax = b com b = {b_plu}:")
print(f"Solução via PLU: x = {x_plu}")
print(f"Resíduo ||Ax - b||_2 = {np.linalg.norm(A_plu @ x_plu - b_plu):.2e}")

# ==============================================================================
# 4. Algoritmo de Thomas para Sistemas Tridiagonais

# ==============================================================================
# Q4.1. — Resolução do sistema tridiagonal 5x5 e verificação do resíduo
# ==============================================================================

print("\n" + "=" * 70)
print("Q4.1. — Algoritmo de Thomas: sistema tridiagonal 5x5")
print("=" * 70)

# Diagonais do sistema tridiagonal 5x5
b_trid = np.array([4, 4, 4, 4, 4], dtype=float)   # diagonal principal
a_trid = np.array([-1, -1, -1, -1], dtype=float)   # subdiagonal (n-1 elementos)
c_trid = np.array([-1, -1, -1, -1], dtype=float)   # superdiagonal (n-1 elementos)
d_trid = np.array([1, 0, 0, 0, 1], dtype=float)    # lado direito

x_thomas = thomas(a_trid, b_trid, c_trid, d_trid)

# Monta a matriz densa para verificar o resíduo
A_trid = montar_tridiagonal(a_trid, b_trid, c_trid)

residuo_thomas = np.linalg.norm(A_trid @ x_thomas - d_trid)

print(f"\nSolução via Thomas: {x_thomas}")
print(f"Resíduo ||Ax - d||_2 = {residuo_thomas:.2e}")
print("\nObservação:")
print("Este sistema surge na discretização por diferenças finitas de")
print("equações diferenciais ordinárias de 2ª ordem (problema de valor")
print("de contorno), como a equação de Poisson 1D: -u'' = f.")

# ==============================================================================
# Q4.2. — Thomas vs Gauss: escalonamento em função de n
# ==============================================================================

print("\n" + "=" * 70)
print("Q4.2. — Thomas vs Gauss: escalonamento (tempo × n)")
print("=" * 70)

ns_thomas = [100, 500, 1000, 5000, 10000]

tempos_thomas = []
tempos_gauss  = []

print(f"\n{'n':>6}  {'Thomas (ms)':>12}  {'Gauss (ms)':>14}  {'Razão':>8}  {'Razão teórica':>14}")
print("-" * 62)

for n_t in ns_thomas:
    # Gera sistema tridiagonal: bi=4, ai=ci=-1, di=1
    b_t = np.full(n_t, 4.0)
    a_t = np.full(n_t - 1, -1.0)
    c_t = np.full(n_t - 1, -1.0)
    d_t = np.ones(n_t)

    # Tempo Thomas
    t0 = time.perf_counter()
    thomas(a_t, b_t, c_t, d_t)
    t_th = (time.perf_counter() - t0) * 1000  # ms

    # Tempo Gauss (matriz densa) — só executa para n pequeno para não travar
    if n_t <= 1000:
        A_dense = montar_tridiagonal(a_t, b_t, c_t)
        t0 = time.perf_counter()
        resolver_gauss(A_dense, d_t.copy())
        t_gs = (time.perf_counter() - t0) * 1000  # ms
    else:
        t_gs = float('nan')  # inviável para n grande

    tempos_thomas.append(t_th)
    tempos_gauss.append(t_gs)

    razao_obs  = t_gs / t_th if not np.isnan(t_gs) else float('nan')
    # Razão teórica de flops: Gauss usa ~2n³/3, Thomas usa ~8n => razão ≈ n²/12.
    # A razão observada é menor porque ambas são implementações Python puro,
    # com constantes multiplicativas próximas — o overhead de laço domina para n pequeno.
    razao_teor = n_t**2 / 12

    if np.isnan(t_gs):
        print(f"{n_t:>6}  {t_th:>12.4f}  {'N/A (inviável)':>14}  {'N/A':>8}  {razao_teor:>14.1f}")
    else:
        print(f"{n_t:>6}  {t_th:>12.4f}  {t_gs:>14.4f}  {razao_obs:>8.1f}  {razao_teor:>14.1f}")

# Gráfico Thomas vs Gauss (apenas n onde Gauss foi executado)
ns_plot  = [n  for n,  tg in zip(ns_thomas, tempos_gauss) if not np.isnan(tg)]
tg_plot  = [tg for tg in tempos_gauss if not np.isnan(tg)]
tth_plot = tempos_thomas[:len(ns_plot)]

plt.figure()
plt.loglog(ns_plot, tth_plot, 'o-', label='Thomas O(n)')
plt.loglog(ns_plot, tg_plot,  's-', label='Gauss O(n³)')
plt.xlabel('n')
plt.ylabel('Tempo (ms)')
plt.title('Thomas vs Gauss: tempo de solução × n')
plt.legend(frameon=False)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("thomas_vs_gauss.pdf", dpi=150)

print("\nObservação sobre a razão teórica vs observada:")
print("A razão teórica (n²/12) assume que Gauss custa ~2n³/3 flops e Thomas")
print("~8n flops. A razão observada é menor porque ambas as implementações")
print("são Python puro com laços explícitos: o overhead de interpretação")
print("afeta os dois métodos de forma parecida, reduzindo a razão real.")

# ==============================================================================
# Q4.3. — Memória: matriz densa vs representação tridiagonal (n=10000)
# ==============================================================================

print("\n" + "=" * 70)
print("Q4.3. — Uso de memória: densa vs tridiagonal (n = 10 000)")
print("=" * 70)

n_mem = 10_000

mem_densa_mb    = n_mem**2 * 8 / 2**20
mem_tridiagl_mb = 3 * n_mem * 8 / 2**20   # três vetores de comprimento n

print(f"\nMatriz densa ({n_mem}×{n_mem}, float64): {mem_densa_mb:,.1f} MB")
print(f"Representação tridiagonal (3 vetores):  {mem_tridiagl_mb:.4f} MB")
print(f"Razão densa / tridiagonal: {mem_densa_mb / mem_tridiagl_mb:,.0f}×")
print("\nImplicação: armazenar e fatorar a matriz densa para n=10 000 é")
print("inviável na prática (~762 MB só para A). O Algoritmo de Thomas")
print("resolve o mesmo sistema com menos de 1 MB e em tempo linear.")

# ==============================================================================

# TODO: 3. Fatoração de Cholesky para Matrizes SPD
# TODO: 5. Custo Computacional Empírico
# TODO: 6. Condicionamento e Sensibilidade à Perturbação
# TODO: 7. Projeto Integrador: PageRank Numérico
# TODO: 8. Desafio (Opcional — Pontuação Extra)

plt.show()