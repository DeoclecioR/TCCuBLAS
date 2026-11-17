# ---------------------------------------------------------------------------
# Imagem base da NVIDIA: traz o toolkit CUDA + cuBLAS + nvcc.
# Use a tag "devel" (e nao "runtime"), pois precisamos compilar codigo C/CUDA.
#
# IMPORTANTE: a versao de CUDA abaixo deve ser compativel com o DRIVER NVIDIA
# instalado na maquina host. Verifique seu driver com `nvidia-smi` e ajuste a
# tag se necessario. A GTX 1050 Ti (Pascal, compute capability 6.1) e suportada
# pelo CUDA 12.x.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Evita prompts interativos durante o apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Arquitetura da GPU alvo, usada na compilacao (sm_61 = GTX 1050 Ti / Pascal).
# Se for rodar em outra GPU, ajuste aqui (ex.: sm_75 para Turing, sm_86 para Ampere).
ARG CUDA_ARCH=sm_61

# ---------------------------------------------------------------------------
# Dependencias de sistema: Python, pip e ferramentas de build.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        make \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Dependencias Python (NumPy e CuPy para o harness de comparacao).
# Copiamos so o requirements primeiro para aproveitar o cache de camadas:
# enquanto o requirements nao mudar, esta etapa nao e refeita.
# ---------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Copia o codigo-fonte do projeto.
# ---------------------------------------------------------------------------
COPY . .

# ---------------------------------------------------------------------------
# Compila a biblioteca C/CUDA como objeto compartilhado (.so).
# O contexto de build e a pasta TCCuBLAS/: o Makefile gera aqui o libTCCuBLAS.so
# que o Python (tccublas.py) carrega via ctypes.
# Passamos a arquitetura da GPU como variavel, para o nvcc gerar codigo correto.
# ---------------------------------------------------------------------------
RUN make CUDA_ARCH=${CUDA_ARCH}

# ---------------------------------------------------------------------------
# Comando padrao: executa o harness de benchmark. Os resultados sao gravados em
# /app/resultados (montado como volume pelo docker-compose.yml).
# ---------------------------------------------------------------------------
CMD ["python3", "auxiliares/run_benchmarks.py"]
