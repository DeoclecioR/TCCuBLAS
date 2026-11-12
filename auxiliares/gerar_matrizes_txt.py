"""
Gera duas matrizes densas float32 e as salva em arquivos TXT para leitura em C.

Uso pela linha de comando:
    python3 gerar_matrizes_txt.py LINHAS_A COLUNAS_A LINHAS_B COLUNAS_B [--seed 42]
"""

import argparse

import numpy as np


def salvar_matriz_txt(matriz, caminho):
    """Grava uma matriz 2D no formato descrito acima."""
    linhas, colunas = matriz.shape
    with open(caminho, "w", encoding="ascii", newline="\n") as f:
        f.write(f"{linhas} {colunas}\n")
        for linha in matriz:
            f.write(" ".join(f"{v:.9g}" for v in linha) + "\n")


def gerar_matrizes(linhas_a, colunas_a, linhas_b, colunas_b, seed=42,
                   caminho_a="matriz_a.txt", caminho_b="matriz_b.txt"):
    """
    Gera duas matrizes densas float32 (valores uniformes em [0, 1)) e as salva
    em TXT.


    Retorna (A, B) como arrays NumPy float32.
    """
    for nome, valor in (("linhas_a", linhas_a), ("colunas_a", colunas_a),
                        ("linhas_b", linhas_b), ("colunas_b", colunas_b)):
        if not isinstance(valor, (int, np.integer)) or valor < 1:
            raise ValueError(f"{nome} deve ser um inteiro >= 1 (recebido: {valor!r})")

    rng = np.random.default_rng(seed)
    a = rng.random((linhas_a, colunas_a), dtype=np.float32)
    b = rng.random((linhas_b, colunas_b), dtype=np.float32)

    salvar_matriz_txt(a, caminho_a)
    salvar_matriz_txt(b, caminho_b)
    return a, b


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("linhas_a", type=int)
    p.add_argument("colunas_a", type=int)
    p.add_argument("linhas_b", type=int)
    p.add_argument("colunas_b", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--saida-a", default="matriz_a.txt")
    p.add_argument("--saida-b", default="matriz_b.txt")
    args = p.parse_args()

    gerar_matrizes(args.linhas_a, args.colunas_a, args.linhas_b, args.colunas_b,
                   args.seed, args.saida_a, args.saida_b)
    print(f"A ({args.linhas_a}x{args.colunas_a}) -> {args.saida_a}")
    print(f"B ({args.linhas_b}x{args.colunas_b}) -> {args.saida_b}")


if __name__ == "__main__":
    main()
