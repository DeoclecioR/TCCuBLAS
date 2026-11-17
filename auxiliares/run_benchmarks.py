"""
Harness de comparacao de desempenho.

Compara, sobre as MESMAS cadeias de matrizes, quatro abordagens:
  1. Encadeamento simples (ordem ingenua, esquerda->direita) em NumPy (CPU)
  2. numpy.multi_dot (CPU, ordem otima)
  3. CuPy (GPU, ordem otima via DP + cp.matmul; CuPy nao tem multi_dot)
  4. A biblioteca desenvolvida (TCCuBLAS, via TCCuBLAS/tccublas.py);

Para cada caso de teste, aplica aquecimento (descarta as primeiras iterações),
repete a medição varias vezes e reporta mediana e desvio.

Metricas coletadas por abordagem e caso:
  - tempo total ponta a ponta (mediana e desvio)
  - tempos por etapa: transferencia host->device, calculo, device->host
    (GPU: CuPy medido no Python; biblioteca medida dentro do C)
  - razao transferencia/calculo
  - numero de multiplicacoes escalares (ordem otima e ordem ingenua) e o
    ganho algoritmico (ingenua / otima), independentes de hardware
  - FLOPs e GFLOPS (sobre o tempo de calculo)
  - erro numerico maximo (absoluto e relativo) contra a referencia
  - uso de memoria da GPU (bytes)

Escreve os resultados em TCCuBLAS/resultados/benchmarks.csv (e ambiente.txt),
qualquer que seja a pasta de onde o script for executado.



"""

import csv
import os
import statistics
import sys
import time
from functools import reduce

import numpy as np

PASTA_TCC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PASTA_TCC not in sys.path:
    sys.path.insert(0, PASTA_TCC)
PASTA_RESULTADOS = os.path.join(PASTA_TCC, "resultados")

from generator import (
    gerar_dimensoes,
    gerar_cadeia,
    custo_parentizacao_otima,
    custo_ordem_ingenua,
    flops_da_ordem,
    tabela_divisoes,
)


try:
    import cupy as cp
    _TEM_CUPY = True
except Exception:
    _TEM_CUPY = False


try:
    from tccublas import multiplicar_cadeia as tcc_multiplicar_cadeia
    _TEM_LIB = True
except Exception as e:
    print(f"[aviso] biblioteca TCCuBLAS nao carregada: {e}")
    _TEM_LIB = False


SEED = 42           # Semente pseudoaleatória
N_WARMUP = 3        # iteracoes de aquecimento descartadas
N_REPETICOES = 10   # medicoes efetivas por caso
TOLERANCIA = 1e-3   # tolerancia relativa (float32 tem ~1e-7 de epsilon; acumula na cadeia)

# Cada caso: (numero de matrizes, dimensao minima, dimensao maxima)
CASOS = [
    (4, 50, 400),
    (6, 50, 400),
    (8, 50, 400),
]

ETAPAS = ("h2d_s", "calculo_s", "d2h_s")

# Abordagens que multiplicam a cadeia da esquerda para a direita (custo da
# ordem ingenua, usado no calculo de FLOPs/GFLOPS).
ORDEM_INGENUA = {"encadeamento_simples", "biblioteca_desenvolvida"}

# GPU alvo: GTX 1050 Ti com 4 GiB. Reserva ~0,5 GiB para contexto CUDA/driver/
# display e limita o uso estimado a 3,5 GiB. Dados em float32 (4 bytes).
BYTES_FLOAT32 = 4
VRAM_LIMITE_BYTES = int(3.5 * 1024**3)


def estimar_vram_bytes(dims):
    """
    Estimativa conservadora (float32) da VRAM necessaria: todas as matrizes de
    entrada + todos os intermediarios possiveis (dims[0] x dims[k]) + o
    resultado, com folga de 2x para o pool do CuPy/workspace do cuBLAS.
    """
    n = len(dims) - 1
    entradas = sum(dims[i] * dims[i + 1] for i in range(n))
    intermediarios = sum(dims[0] * dims[k] for k in range(2, n + 1))
    return 2 * BYTES_FLOAT32 * (entradas + intermediarios)


def registrar_ambiente(caminho):
    # Grava o ambiente de execucao (GPU, driver, versoes) junto aos resultados, para reprodução posterior
    import platform
    import subprocess
    linhas = [f"python={platform.python_version()}", f"numpy={np.__version__}",
              f"plataforma={platform.platform()}"]
    if _TEM_CUPY:
        linhas.append(f"cupy={cp.__version__}")
    try:
        saida = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        linhas.append(f"gpu,driver,vram={saida}")
    except Exception:
        linhas.append("gpu=indisponivel")
    with open(caminho, "w") as f:
        f.write("\n".join(linhas) + "\n")


# ---------------------------------------------------------------------------
# Abordagens a comparar. Cada uma recebe a lista de matrizes e devolve
# (produto como array NumPy, dicionario de tempos por etapa em segundos).
# O dicionario e vazio quando a abordagem nao separa etapas (CPU pura).
# ---------------------------------------------------------------------------
def encadeamento_simples(matrices):
    """Ordem ingenua: multiplica da esquerda para a direita."""
    return reduce(np.matmul, matrices), {}


def numpy_multidot(matrices):
    return np.linalg.multi_dot(matrices), {}


def cupy_multidot(matrices):
    """
    Transfere para a GPU, multiplica com cupy.multi_dot e traz de volta,
    cronometrando cada etapa. A sincronizacao antes de cada leitura de relogio
    é necessaria porque a execucao em GPU é assincrona.
    """
    # CuPy nao possui multi_dot: usamos a ordem otima (DP em Python) e cp.matmul.
    dims = [matrices[0].shape[0]] + [m.shape[1] for m in matrices]
    splits = tabela_divisoes(dims)

    def multiplicar(i, j):
        if i == j:
            return gpu_mats[i]
        k = splits[i][j]
        return cp.matmul(multiplicar(i, k), multiplicar(k + 1, j))

    t0 = time.perf_counter()
    gpu_mats = [cp.asarray(m) for m in matrices]
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()

    resultado_gpu = multiplicar(0, len(gpu_mats) - 1)
    cp.cuda.Stream.null.synchronize()
    t2 = time.perf_counter()

    resultado = cp.asnumpy(resultado_gpu)
    t3 = time.perf_counter()

    return resultado, {"h2d_s": t1 - t0, "calculo_s": t2 - t1, "d2h_s": t3 - t2}


def biblioteca_desenvolvida(matrices):
    
    #TCCuBLAS: tempos por etapa medidos DENTRO da biblioteca C (ms -> s).
   
    
    resultado, t = tcc_multiplicar_cadeia(matrices)
    return resultado, {
        "h2d_s": t["h2d_ms"] / 1e3,
        "calculo_s": t["calculo_ms"] / 1e3,
        "d2h_s": t["d2h_ms"] / 1e3,
    }


def cronometrar(func, matrices, n_warmup, n_repeticoes):
    """
    Executa aquecimento e depois mede n_repeticoes vezes.
    Retorna (tempos totais em s, resultado da ultima execucao, lista de
    dicionarios de etapas, um por repeticao).
    """
    # Aquecimento: resultados descartados.
    for _ in range(n_warmup):
        func(matrices)

    tempos = []
    etapas = []
    resultado = None
    for _ in range(n_repeticoes):
        inicio = time.perf_counter()
        resultado, e = func(matrices)
        fim = time.perf_counter()
        tempos.append(fim - inicio)
        etapas.append(e)
    return tempos, resultado, etapas


def mediana_etapa(etapas, chave):
    """Mediana de uma etapa entre as repeticoes, ou None se nao existir."""
    valores = [e[chave] for e in etapas if chave in e]
    return statistics.median(valores) if valores else None


def memoria_gpu_bytes(nome):
    
    #Uso de memoria da GPU da abordagem, em bytes ("None" se nao se aplica).
    if nome == "cupy_multidot" and _TEM_CUPY:
        return cp.get_default_memory_pool().total_bytes()
    return None


def vazio_se_none(valor):
    return "" if valor is None else valor


def main():
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    linhas_csv = []

    # Monta a lista de abordagens disponiveis neste ambiente.
    abordagens = [
        ("encadeamento_simples", encadeamento_simples),
        ("numpy_multidot", numpy_multidot),
    ]
    if _TEM_CUPY:
        abordagens.append(("cupy_multidot", cupy_multidot))
    else:
        print("[aviso] CuPy indisponivel -- pulando medicoes de GPU de referencia.")
    if _TEM_LIB:
        abordagens.append(("biblioteca_desenvolvida", biblioteca_desenvolvida))

    for n, dmin, dmax in CASOS:
        dims = gerar_dimensoes(n, dmin, dmax, SEED)

        vram_estimada = estimar_vram_bytes(dims)
        if vram_estimada > VRAM_LIMITE_BYTES:
            print(f"[aviso] caso n={n} pulado: VRAM estimada "
                  f"{vram_estimada / 1024**3:.2f} GiB > limite "
                  f"{VRAM_LIMITE_BYTES / 1024**3:.2f} GiB (GTX 1050 Ti 4 GB).")
            continue

        matrices = gerar_cadeia(dims, SEED)

        # Custos independentes de hardware (calculados, nao cronometrados).
        custo_otimo = custo_parentizacao_otima(dims)
        custo_ingenuo = custo_ordem_ingenua(dims)
        ganho_algoritmico = custo_ingenuo / custo_otimo if custo_otimo else None

        print(f"\n=== Caso n={n} dims={dims} ===")
        print(f"Custo otimo (DP em Python)   : {custo_otimo}")
        print(f"Custo ingenuo (esq->dir)     : {custo_ingenuo}")
        if ganho_algoritmico is not None:
            print(f"Ganho algoritmico (ing/otim) : {ganho_algoritmico:.2f}x")

        # Referencia de corretude: resultado do numpy.multi_dot.
        referencia, _ = numpy_multidot(matrices)
        assert referencia.dtype == np.float32, "cadeia deve ser float32"
        escala_ref = float(np.max(np.abs(referencia))) or 1.0

        for nome, func in abordagens:
            # Zera o pool do CuPy para a medicao de memoria ser desta abordagem.
            if _TEM_CUPY:
                cp.get_default_memory_pool().free_all_blocks()

            tempos, resultado, etapas = cronometrar(
                func, matrices, N_WARMUP, N_REPETICOES
            )

            # Erro numerico contra a referencia (absoluto e relativo).
            erro_abs = float(np.max(np.abs(resultado - referencia)))
            erro_rel = erro_abs / escala_ref
            correto = bool(resultado.dtype == np.float32
                           and np.allclose(resultado, referencia,
                                           rtol=TOLERANCIA, atol=TOLERANCIA * escala_ref))

            # Tempos: total e por etapa.
            total_mediana = statistics.median(tempos)
            total_desvio = statistics.pstdev(tempos) if len(tempos) > 1 else 0.0
            h2d = mediana_etapa(etapas, "h2d_s")
            calculo = mediana_etapa(etapas, "calculo_s")
            d2h = mediana_etapa(etapas, "d2h_s")

            
            tempo_calculo = calculo if calculo is not None else total_mediana

            
            razao = None
            if h2d is not None and d2h is not None and tempo_calculo > 0:
                razao = (h2d + d2h) / tempo_calculo

            
            custo_usado = custo_ingenuo if nome in ORDEM_INGENUA else custo_otimo
            flops = flops_da_ordem(custo_usado)
            gflops = flops / tempo_calculo / 1e9 if tempo_calculo > 0 else None

            vram = memoria_gpu_bytes(nome)

            print(f"  {nome:26s} total={total_mediana:.6f}s "
                  f"(desvio {total_desvio:.6f}s) "
                  f"calculo={tempo_calculo:.6f}s "
                  f"GFLOPS={gflops:.2f} "
                  f"erro_rel={erro_rel:.2e} correto={correto}")

            linhas_csv.append({
                "caso_n": n,
                "dims": "x".join(map(str, dims)),
                "abordagem": nome,
                "total_mediana_s": total_mediana,
                "total_desvio_s": total_desvio,
                "h2d_s": vazio_se_none(h2d),
                "calculo_s": tempo_calculo,
                "d2h_s": vazio_se_none(d2h),
                "razao_transf_calculo": vazio_se_none(razao),
                "custo_otimo_mults": custo_otimo,
                "custo_ingenuo_mults": custo_ingenuo,
                "ganho_algoritmico": vazio_se_none(ganho_algoritmico),
                "flops": flops,
                "gflops": vazio_se_none(gflops),
                "erro_abs_max": erro_abs,
                "erro_rel_max": erro_rel,
                "correto": correto,
                "vram_bytes": vazio_se_none(vram),
            })

    # Grava o CSV com os resultados brutos (parte do artefato de reprodutibilidade).
    caminho = os.path.join(PASTA_RESULTADOS, "benchmarks.csv")
    campos = [
        "caso_n", "dims", "abordagem",
        "total_mediana_s", "total_desvio_s",
        "h2d_s", "calculo_s", "d2h_s", "razao_transf_calculo",
        "custo_otimo_mults", "custo_ingenuo_mults", "ganho_algoritmico",
        "flops", "gflops",
        "erro_abs_max", "erro_rel_max", "correto",
        "vram_bytes",
    ]
    # Formato pt-BR (abre direto no Excel) Para ler com pandas: pd.read_csv(caminho, sep=";", decimal=",").
    def formatar(valor):
        if isinstance(valor, float):
            return repr(valor).replace(".", ",")
        return valor

    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, delimiter=";")
        writer.writeheader()
        for linha in linhas_csv:
            writer.writerow({k: formatar(v) for k, v in linha.items()})

    registrar_ambiente(os.path.join(PASTA_RESULTADOS, "ambiente.txt"))
    print(f"\nResultados gravados em {caminho}")


if __name__ == "__main__":
    main()