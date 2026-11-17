"""
Testes do binding Python da TCCuBLAS (sem dependencia de pytest).

Execucao (com a biblioteca ja compilada nesta pasta):
    python teste_binding.py

Cada teste compara a TCCuBLAS com o NumPy ou verifica o comportamento esperado
(erros, entradas nao alteradas, conversao de tipo). Sai com codigo != 0 se algum
teste falhar.
"""

import os
import sys
import tempfile
import traceback

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "auxiliares"))

import tccublas  # noqa: E402
from gerar_matrizes_txt import gerar_matrizes  # noqa: E402

RNG = np.random.default_rng(123)


def aleatoria(linhas, colunas, faixa=(0.0, 1.0)):
    lo, hi = faixa
    return (lo + (hi - lo) * RNG.random((linhas, colunas))).astype(np.float32)


def confere(obtido, esperado, nome):
    assert obtido.dtype == np.float32, f"{nome}: dtype {obtido.dtype}"
    assert obtido.shape == esperado.shape, f"{nome}: forma {obtido.shape} != {esperado.shape}"
    escala = max(1.0, float(np.abs(esperado).max()))
    assert np.allclose(obtido, esperado, rtol=1e-5, atol=1e-5 * escala), (
        f"{nome}: erro maximo {np.abs(obtido - esperado).max():.3e}")


# ---------------------------------------------------------------------------
def teste_soma_e_subtracao():
    for forma in [(1, 1), (1, 9), (9, 1), (7, 13), (50, 3), (128, 128), (300, 257)]:
        a, b = aleatoria(*forma), aleatoria(*forma, faixa=(-1.0, 1.0))
        r, _ = tccublas.somar(a, b)
        confere(r, a + b, f"soma {forma}")
        r, _ = tccublas.subtrair(a, b)
        confere(r, a - b, f"subtracao {forma}")


def teste_multiplicacao():
    for m, k, n in [(1, 1, 1), (1, 9, 1), (3, 4, 2), (7, 13, 5), (120, 300, 80),
                    (257, 129, 63), (500, 500, 500)]:
        a, b = aleatoria(m, k), aleatoria(k, n)
        r, _ = tccublas.multiplicar(a, b)
        confere(r, a @ b, f"multiplicacao {(m, k, n)}")


def teste_valores_negativos():
    a, b = aleatoria(40, 30, (-5, 5)), aleatoria(30, 20, (-5, 5))
    confere(tccublas.multiplicar(a, b)[0], a @ b, "multiplicacao negativos")
    confere(tccublas.subtrair(a, a)[0], np.zeros_like(a), "a - a")


def teste_entradas_nao_sao_alteradas():
    a, b = aleatoria(20, 30), aleatoria(30, 10)
    a0, b0 = a.copy(), b.copy()
    tccublas.multiplicar(a, b)
    tccublas.somar(a, a)
    assert np.array_equal(a, a0) and np.array_equal(b, b0), "entrada foi alterada"


def teste_conversao_de_tipo_e_layout():
    a64 = RNG.random((15, 25))                     # float64
    b64 = RNG.random((25, 11))
    r, _ = tccublas.multiplicar(a64, b64)
    confere(r, (a64.astype(np.float32) @ b64.astype(np.float32)), "float64 -> float32")

    at = aleatoria(25, 15).T                       # nao contiguo (transposta)
    assert not at.flags["C_CONTIGUOUS"]
    b = aleatoria(25, 7)
    confere(tccublas.multiplicar(at, b)[0], np.ascontiguousarray(at) @ b, "nao contiguo")

    lista = [[1, 2, 3], [4, 5, 6]]                 # lista Python de inteiros
    confere(tccublas.somar(lista, lista)[0], np.array(lista, np.float32) * 2, "lista")


def teste_erros():
    a, b = aleatoria(3, 4), aleatoria(5, 2)
    for funcao in (tccublas.somar, tccublas.subtrair, tccublas.multiplicar):
        try:
            funcao(a, b)
        except tccublas.TCCudaErro as e:
            assert e.codigo == 3, f"codigo {e.codigo} em vez de 3 (dimensoes)"
        else:
            raise AssertionError(f"{funcao.__name__} aceitou dimensoes incompativeis")

    for invalida in (np.zeros(5), np.zeros((2, 3, 4)), np.zeros((0, 3))):
        try:
            tccublas.somar(invalida, invalida)
        except ValueError:
            pass
        else:
            raise AssertionError(f"forma invalida aceita: {np.shape(invalida)}")


def teste_tempos():
    a, b = aleatoria(200, 200), aleatoria(200, 200)
    _, t = tccublas.multiplicar(a, b)
    assert set(t) == {"h2d_ms", "calculo_ms", "d2h_ms", "total_ms"}
    assert all(v >= 0 for v in t.values()), t
    soma = t["h2d_ms"] + t["calculo_ms"] + t["d2h_ms"]
    assert abs(soma - t["total_ms"]) < 1e-6, (soma, t["total_ms"])


def teste_cadeia():
    for dims in [(5, 5), (4, 6, 3), (8, 3, 9, 2, 7), (30, 40, 20, 50, 10, 25)]:
        mats = [aleatoria(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        esperado = mats[0]
        for m in mats[1:]:
            esperado = esperado @ m
        r, t = tccublas.multiplicar_cadeia(mats)
        confere(r, esperado, f"cadeia {dims}")
        assert t["total_ms"] >= 0

    unica = aleatoria(6, 4)
    r, t = tccublas.multiplicar_cadeia([unica])
    assert np.array_equal(r, unica) and t["total_ms"] == 0.0
    assert r is not unica, "deve devolver copia"

    try:
        tccublas.multiplicar_cadeia([])
    except ValueError:
        pass
    else:
        raise AssertionError("cadeia vazia aceita")


def teste_leitura_de_arquivo_pela_biblioteca():
    with tempfile.TemporaryDirectory() as d:
        pa, pb = os.path.join(d, "a.txt"), os.path.join(d, "b.txt")
        a, b = gerar_matrizes(17, 9, 9, 23, 5, pa, pb)
        assert np.array_equal(tccublas.ler_matriz(pa), a), "leitura de A difere"
        assert np.array_equal(tccublas.ler_matriz(pb), b), "leitura de B difere"
        # Matriz que veio do arquivo tambem entra nas operacoes.
        confere(tccublas.multiplicar(tccublas.ler_matriz(pa), tccublas.ler_matriz(pb))[0],
                a @ b, "multiplicacao de matrizes lidas de arquivo")

        try:
            tccublas.ler_matriz(os.path.join(d, "nao_existe.txt"))
        except tccublas.TCCudaErro as e:
            assert e.codigo == 2, e.codigo
        else:
            raise AssertionError("arquivo inexistente aceito")

        ruim = os.path.join(d, "ruim.txt")
        with open(ruim, "w") as f:
            f.write("2 2\n1 2 3\n")                # faltam valores
        try:
            tccublas.ler_matriz(ruim)
        except tccublas.TCCudaErro as e:
            assert e.codigo == 2, e.codigo
        else:
            raise AssertionError("arquivo truncado aceito")


def teste_chamadas_repetidas():
    # Muitas chamadas seguidas: sem travar e sem resultados diferentes.
    a, b = aleatoria(64, 64), aleatoria(64, 64)
    esperado = a @ b
    for i in range(100):
        r, _ = tccublas.multiplicar(a, b)
        confere(r, esperado, f"repeticao {i}")


# ---------------------------------------------------------------------------
def main():
    testes = [(n, f) for n, f in sorted(globals().items())
              if n.startswith("teste_") and callable(f)]
    falhas = 0
    for nome, funcao in testes:
        try:
            funcao()
            print(f"[ok]    {nome}")
        except Exception:
            falhas += 1
            print(f"[FALHA] {nome}")
            traceback.print_exc()
    print(f"\n{len(testes) - falhas}/{len(testes)} testes passaram")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
