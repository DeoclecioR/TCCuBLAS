"""
Binding Python (ctypes) para a biblioteca TCCuBLAS.

Recebe matrizes NumPy (convertidas para float32 C-contiguo), passa-as para a
biblioteca C/CUDA e devolve o resultado como array NumPy float32, junto com os
tempos medidos dentro da biblioteca (cudaEvent, em ms).

Funcoes:
    somar(a, b)        -> (a + b, tempos)
    subtrair(a, b)     -> (a - b, tempos)
    multiplicar(a, b)  -> (a @ b, tempos)
    multiplicar_cadeia(matrizes) -> (A1 @ A2 @ ... @ An, tempos)
    ler_matriz(caminho) -> array lido de um TXT pela propria biblioteca

`tempos` e um dicionario com h2d_ms, calculo_ms, d2h_ms e total_ms.

Erros da biblioteca viram `TCCudaErro` (atributo `codigo`).

NOTA: multiplicar_cadeia multiplica da esquerda para a direita (sem ordem
otima) e a cada passo o resultado parcial volta ao host e e enviado de novo
para a GPU. A ordem otima e a cadeia residente na GPU ficam para a proxima
etapa do trabalho.
"""

import ctypes
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    # Python 3.8+ nao usa o PATH para achar DLLs dependentes (cublas64_*.dll):
    # registra a pasta bin do CUDA explicitamente.
    _cuda = os.environ.get("CUDA_PATH")
    if _cuda and os.path.isdir(os.path.join(_cuda, "bin")):
        os.add_dll_directory(os.path.join(_cuda, "bin"))
    _CAMINHO = os.path.join(_DIR, "libTCCuBLAS.dll")
else:
    _CAMINHO = os.path.join(_DIR, "libTCCuBLAS.so")

_lib = ctypes.CDLL(_CAMINHO)


# ---------------------------------------------------------------------------
# Estruturas identicas as de TCCuBLAS.h
# ---------------------------------------------------------------------------
class _Matriz(ctypes.Structure):
    _fields_ = [
        ("linhas", ctypes.c_int),
        ("colunas", ctypes.c_int),
        ("dados", ctypes.POINTER(ctypes.c_float)),
    ]


class _Tempos(ctypes.Structure):
    _fields_ = [
        ("h2d_ms", ctypes.c_double),
        ("calculo_ms", ctypes.c_double),
        ("d2h_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
    ]


_PT_MATRIZ = ctypes.POINTER(_Matriz)

for _nome in ("TCCuda_somar", "TCCuda_subtrair", "TCCuda_multiplicar"):
    _f = getattr(_lib, _nome)
    _f.argtypes = [_PT_MATRIZ, _PT_MATRIZ,
                   ctypes.POINTER(_PT_MATRIZ), ctypes.POINTER(_Tempos)]
    _f.restype = ctypes.c_int

_lib.TCCuda_ler_matriz.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
_lib.TCCuda_ler_matriz.restype = _PT_MATRIZ

_lib.TCCuda_liberar_matriz.argtypes = [_PT_MATRIZ]
_lib.TCCuda_liberar_matriz.restype = None

_lib.TCCuda_mensagem_erro.argtypes = [ctypes.c_int]
_lib.TCCuda_mensagem_erro.restype = ctypes.c_char_p


class TCCudaErro(RuntimeError):
    """Erro devolvido pela biblioteca TCCuBLAS (codigo em `codigo`)."""

    def __init__(self, codigo):
        self.codigo = int(codigo)
        texto = _lib.TCCuda_mensagem_erro(self.codigo).decode("utf-8", "replace")
        super().__init__(f"TCCuBLAS: {texto} (codigo {self.codigo})")


# ---------------------------------------------------------------------------
# Conversoes
# ---------------------------------------------------------------------------
def _preparar(matriz):
    """Converte para um array 2D float32 C-contiguo (sem alterar o original)."""
    a = np.ascontiguousarray(matriz, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError(f"a matriz deve ser 2D e nao vazia (forma recebida: {a.shape})")
    return a


def _estrutura(array):
    """Estrutura TCCudaMatriz apontando para os dados do array (sem copiar)."""
    return _Matriz(array.shape[0], array.shape[1],
                   array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))


def _copiar_e_liberar(ptr):
    """Copia a matriz alocada pela biblioteca para um array NumPy e a libera."""
    try:
        m = ptr.contents
        return np.ctypeslib.as_array(m.dados, shape=(m.linhas, m.colunas)).copy()
    finally:
        _lib.TCCuda_liberar_matriz(ptr)


def _dicionario(t):
    return {"h2d_ms": t.h2d_ms, "calculo_ms": t.calculo_ms,
            "d2h_ms": t.d2h_ms, "total_ms": t.total_ms}


def _operar(funcao, a, b):
    a = _preparar(a)
    b = _preparar(b)
    ma, mb = _estrutura(a), _estrutura(b)   # a e b permanecem vivos ate o fim
    resultado = _PT_MATRIZ()
    tempos = _Tempos()
    rc = funcao(ctypes.byref(ma), ctypes.byref(mb),
                ctypes.byref(resultado), ctypes.byref(tempos))
    if rc != 0:
        raise TCCudaErro(rc)
    return _copiar_e_liberar(resultado), _dicionario(tempos)


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------
def somar(a, b):
    """a + b na GPU. a e b precisam ter a mesma forma."""
    return _operar(_lib.TCCuda_somar, a, b)


def subtrair(a, b):
    """a - b na GPU. a e b precisam ter a mesma forma."""
    return _operar(_lib.TCCuda_subtrair, a, b)


def multiplicar(a, b):
    """a @ b na GPU. Colunas de a = linhas de b."""
    return _operar(_lib.TCCuda_multiplicar, a, b)


def multiplicar_cadeia(matrizes):
    """
    Produto A1 @ A2 @ ... @ An, da esquerda para a direita, usando
    TCCuda_multiplicar a cada passo. Os tempos devolvidos sao a SOMA dos passos
    (inclui as transferencias dos resultados parciais).
    """
    matrizes = [_preparar(m) for m in matrizes]
    if not matrizes:
        raise ValueError("a cadeia precisa ter ao menos uma matriz")

    acumulado = matrizes[0].copy()
    total = {"h2d_ms": 0.0, "calculo_ms": 0.0, "d2h_ms": 0.0, "total_ms": 0.0}
    for proxima in matrizes[1:]:
        acumulado, t = multiplicar(acumulado, proxima)
        for chave in total:
            total[chave] += t[chave]
    return acumulado, total


def ler_matriz(caminho):
    """Le um TXT (formato de gerar_matrizes_txt.py) usando a biblioteca C."""
    erro = ctypes.c_int(0)
    ptr = _lib.TCCuda_ler_matriz(os.fsencode(caminho), ctypes.byref(erro))
    if not ptr:
        raise TCCudaErro(erro.value)
    return _copiar_e_liberar(ptr)
