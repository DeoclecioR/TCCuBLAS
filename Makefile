# ---------------------------------------------------------------------------
# Makefile da biblioteca TCCuBLAS. Funciona no Windows (nvcc + MSVC, via
# mingw32-make) e em Linux / WSL2 / Docker (nvcc + gcc, via make).
# Alvo de referencia: NVIDIA GeForce GTX 1050 Ti (4 GB, Pascal, sm_61), float32.
#
# Uso:
#   make                      # compila para sm_61 (GTX 1050 Ti)
#   make CUDA_ARCH=sm_75      # compila para outra GPU
#   make clean                # remove artefatos
#
# No Windows rode pelo compilar_windows.bat (localiza o MSVC e chama este
# Makefile) ou a partir de um "x64 Native Tools Command Prompt".
#
# Gera:
#   libTCCuBLAS.{dll|so}   biblioteca compartilhada
#   tccublas{.exe}         executavel de linha de comando
# ---------------------------------------------------------------------------

NVCC := nvcc

# sm_61 = GTX 1050 Ti (compute capability 6.1). Sobrescreva se usar outra GPU.
CUDA_ARCH ?= sm_61

# -O3                       otimizacao agressiva
# -arch=$(CUDA_ARCH)        codigo para a GPU alvo
ifeq ($(OS),Windows_NT)
# MSVC: sem -fPIC; o nvcc pode recusar versoes novas do Visual Studio.
NVCCFLAGS := -O3 -arch=$(CUDA_ARCH) -allow-unsupported-compiler
LIB_EXT := dll
EXE_EXT := .exe
RM := cmd /C del /Q /F
LIB_LIMPEZA := libTCCuBLAS.dll libTCCuBLAS.lib libTCCuBLAS.exp tccublas.exe tccublas.lib tccublas.exp
else
# -Xcompiler -fPIC: codigo independente de posicao (exigido pelo .so)
NVCCFLAGS := -O3 -arch=$(CUDA_ARCH) -Xcompiler -fPIC
LIB_EXT := so
EXE_EXT :=
RM := rm -f
LIB_LIMPEZA := libTCCuBLAS.so tccublas
endif
LDFLAGS := -lcublas

TCC_LIB := libTCCuBLAS.$(LIB_EXT)
TCC_BIN := tccublas$(EXE_EXT)

.DEFAULT_GOAL := all
.PHONY: all clean

all: $(TCC_LIB) $(TCC_BIN)

$(TCC_LIB): TCCuBLAS.cu TCCuBLAS.h
	$(NVCC) $(NVCCFLAGS) --shared -o $@ TCCuBLAS.cu $(LDFLAGS)

$(TCC_BIN): tccublas_cli.cu TCCuBLAS.cu TCCuBLAS.h
	$(NVCC) $(NVCCFLAGS) -o $@ tccublas_cli.cu TCCuBLAS.cu $(LDFLAGS)

clean:
	-$(RM) $(LIB_LIMPEZA)
