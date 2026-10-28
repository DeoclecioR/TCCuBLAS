"""
Gerador de cadeias de matrizes para os experimentos.

Gera cadeias de matrizes densas, pseudoaleatorias, a partir de uma semente
fixa, de modo que as mesmas entradas sejam fornecidas a todas as abordagens
comparadas (encadeamento simples, numpy.multi_dot, cupy.multi_dot e a
biblioteca desenvolvida).

O ponto central e gerar dimensoes HETEROGENEAS: o efeito da ordem de
parentizacao so aparece quando as dimensoes variam ao longo da cadeia.

Tambem concentra os calculos de custo independentes de hardware:
  - custo da ordem otima (DP);
  - custo da ordem ingenua (esquerda -> direita);
  - FLOPs a partir do numero de multiplicacoes escalares.
"""

import numpy as np


def gerar_dimensoes(n, dim_min, dim_max, seed):
    """
    Gera (n+1) dimensoes heterogeneas para uma cadeia de n matrizes.
    A matriz i tera forma dims[i] x dims[i+1].

    n       : numero de matrizes na cadeia.
    dim_min : menor dimensao possivel.
    dim_max : maior dimensao possivel.
    seed    : semente para reprodutibilidade.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(dim_min, dim_max + 1, size=n + 1).tolist()


def gerar_dimensoes_homogeneas(n, dim, seed=None):
    """
    Gera (n+1) dimensoes iguais (caso de controle): todas as matrizes sao
    dim x dim. Aqui a ordem de parentizacao praticamente nao altera o custo,
    servindo de contraste para evidenciar o ganho no caso heterogeneo.
    """
    return [dim] * (n + 1)


def gerar_cadeia(dims, seed):
    """
    Gera a lista de matrizes NumPy (float64) a partir de um vetor de dimensoes.
    A matriz i tem forma dims[i] x dims[i+1].

    Retorna a lista de arrays NumPy.
    """
    rng = np.random.default_rng(seed)
    matrices = []
    for i in range(len(dims) - 1):
        m = rng.random((dims[i], dims[i + 1]), dtype=np.float64)
        matrices.append(m)
    return matrices


def custo_parentizacao_otima(dims):
    """
    Calcula, em Python, o custo minimo (numero de multiplicacoes escalares) da
    cadeia pela programacao dinamica classica. Serve como VERIFICACAO
    INDEPENDENTE do valor retornado pela biblioteca C (mc_optimal_order).

    dims: vetor de (n+1) dimensoes.
    """
    n = len(dims) - 1
    # m[i][j] = custo minimo para multiplicar as matrizes i..j (1-indexado)
    INF = float("inf")
    m = [[0] * (n + 1) for _ in range(n + 1)]

    for comprimento in range(2, n + 1):
        for i in range(1, n - comprimento + 2):
            j = i + comprimento - 1
            m[i][j] = INF
            for k in range(i, j):
                custo = (m[i][k] + m[k + 1][j]
                         + dims[i - 1] * dims[k] * dims[j])
                if custo < m[i][j]:
                    m[i][j] = custo
    return m[1][n]


def custo_ordem_ingenua(dims):
    """
    Custo (numero de multiplicacoes escalares) da ordem ingenua, que multiplica
    da esquerda para a direita: ((A1*A2)*A3)*...*An.

    O acumulador tem sempre dims[0] linhas; ao multiplicar pela matriz k
    (dims[k] x dims[k+1]) o custo e dims[0] * dims[k] * dims[k+1].
    """
    n = len(dims) - 1
    custo = 0
    for k in range(1, n):
        custo += dims[0] * dims[k] * dims[k + 1]
    return custo


def flops_da_ordem(custo_multiplicacoes):
    """
    Converte o numero de multiplicacoes escalares em FLOPs.

    Cada produto m x n por n x p faz m*n*p multiplicacoes e aproximadamente
    m*n*p somas, logo ~ 2 * m*n*p operacoes de ponto flutuante. Use esta
    funcao para derivar FLOPS (operacoes por segundo); para falar do custo que
    a parentizacao minimiza, use a contagem de multiplicacoes.
    """
    return 2 * custo_multiplicacoes