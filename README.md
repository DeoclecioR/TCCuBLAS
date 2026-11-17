# TCCuBLAS

Biblioteca em C/CUDA que realiza operações entre matrizes densas, com
float32 na GPU usando o cuBLAS. Ela lê as matrizes de arquivos TXT, ou recebe via Python Bindings,
e imprime as dimensões do resultado, a matriz e o tempo decorrido.


## Estrutura

| Arquivo | Função |

| `TCCuBLAS.h` | API pública (tipos, constantes e funções) |
| `TCCuBLAS.cu` | Implementação |
| `tccublas_cli.cu` | Programa de linha de comando para executar a biblioteca
| `Makefile` | Compila a biblioteca e o executável (Windows e Linux) |
| `compilar_windows.bat` | No Windows, localiza o MSVC e chama o `Makefile` |
| `auxiliares/gerar_matrizes_txt.py` | Gera duas matrizes float32 em TXT |
| `auxiliares/generator.py`, `auxiliares/run_benchmarks.py` | Cadeias de matrizes de referência e benchmark que compara a TCCuBLAS com NumPy e CuPy |
| `tccublas.py` | Binding Python (`ctypes`) da biblioteca |
| `teste_binding.py` | Testes do binding contra o NumPy |
| `requirements.txt` | Dependências Python fixadas (NumPy 1.26.4 e CuPy 13.3.0) |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Execução do benchmark em container |


## Softwares necessários PARA REPRODUÇÃO DOS RESULTADOS

Uma **GPU NVIDIA** com driver instalado (`nvidia-smi`). Versões
usadas e validadas:

- GPU de referência: **NVIDIA GeForce GTX 1050 Ti, 4 GB** (Pascal, `sm_61`).

**SOFTWARES NECESSÀRIOS**

Softwares e suas versões para **reprodução dos resultados**
| Driver NVIDIA | 582.78 | 
| CUDA Toolkit (`nvcc`, cuBLAS) | 12.6 | 
| Compilador C/C++ | MSVC v143, toolset 14.36 (VS 2022 17.6) | 
| Make | GNU Make 4.4.1 (`mingw32-make`) | 
| Python + NumPy (para gerar as matrizes) | 3.12 + NumPy 1.26.4 | 
| CuPy | 13.3.0


**OBSERVAÇÕES**

- **Não funciona com CUDA 13 em diante, para fins de reprodução.**
- Ele removeu o suporte à arquitetura Pascal (`sm_61`), e o `nvcc` não compila para a GTX 1050 Ti.

- **No Windows, o `nvcc` só compila com o MSVC** (o MinGW não compila instruções CUDA).
Um MSVC recente demais (toolset 14.5x, Visual Studio 2026) pode derrubar o `nvcc` (`cudafe++ died with status 0xC0000409`). 
Instale o componente **MSVC v143 – VS 2022 C++ x64/x86 build tools (v14.36-17.6)** no Visual Studio
Installer, em *Componentes individuais*, junto com a carga *Desenvolvimento para desktop com C++*.
  
 O `compilar_windows.bat` usa automaticamente o toolset `14.4*` ou `14.3*` que encontrar.
- No Windows, o `mingw32-make` só executa o `Makefile`; quem compila é o `nvcc` com o MSVC.


## Como compilar


**Windows** (PowerShell, na pasta do projeto):

```bash
  .\compilar_windows.bat
```

Que gera `libTCCuBLAS.dll` (ou `.so`) e o executável `tccublas.exe`. 
  Para limpar: `make clean` ou `.\compilar_windows.bat clean`.


## Usar o executável (por CLI, apenas para testar a execução da biblioteca desenvolvida antes de implementar o binding em python. Caso queira obter os benchmarks, passe para a próxima sessão)

1. Gere duas matrizes (os quatro inteiros são linhas e colunas de A e de B):
```bash
    python auxiliares/gerar_matrizes_txt.py 1000 500 500 2000 --saida-a matriz_a.txt --saida-b matriz_b.txt`
```
2. Execute a operação:
```bash
   ./tccublas multiplicacao matriz_a.txt matriz_b.txt       
   ./tccublas soma a.txt b.txt --sem-matriz                 # omite a impressão da matriz
```

Operações: `soma`, `subtracao` e `multiplicacao`. Soma e subtração exigem A e B
com as **mesmas dimensões**; a multiplicação exige **colunas de A = linhas de B**.
Sem `--sem-matriz`, a matriz inteira é impressa (evite em matrizes grandes).


Exemplo de saída:
```
Dimensoes da matriz resultante: 3 x 2
Matriz resultante:
1.325453 0.957836
1.564340 1.329594
1.305282 0.734231
Tempo decorrido (GPU):
  transferencia host->device : 0.0195 ms
  calculo                    : 0.0072 ms
  transferencia device->host : 0.0303 ms
  total                      : 0.0569 ms
```

Código de saída do programa: `0` em sucesso; `1` argumento inválido; `2` falha ao
ler o arquivo; `3` dimensões incompatíveis; `4`, `5` e `6` erro de memória do host,
CUDA e cuBLAS.




## Como rodar o benchmark e dependências Python

O benchmark (`auxiliares/run_benchmarks.py`) compara a TCCuBLAS com NumPy e CuPy
sobre as mesmas cadeias de matrizes e grava `resultados/benchmarks.csv` e
`resultados/ambiente.txt` (GPU, driver e versões usadas). Use um ambiente virtual
com as versões fixadas (Python 3.12 no Windows; o CuPy não tem pacote para o python 3.15, até a presente data, 17/11/2026):

**Execute o powershell/CMD na pasta raiz**
```bash
  py -3.12 -m venv .venv   #Cria o .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt #Instala as dependências
  .\.venv\Scripts\python.exe auxiliares\run_benchmarks.py #Executa os benchmarks entre as bibliotecas
```
**Os dois primeiros comandos são separados. O script pode ser executado de qualquer pasta, e a TCCuBLAS precisa estar compilada.**


## Executar com Docker

Não precisa de Python nem de CUDA instalados no computador, só do driver NVIDIA e
do Docker (Docker Desktop com WSL2 no Windows; no Linux, Docker Engine com o NVIDIA
Container Toolkit). Dentro desta pasta execute por terminal:

```bash
docker compose up --build
```

A imagem (CUDA 12.3.1, Ubuntu 22.04, Python 3.10) compila a biblioteca e roda o
benchmark. Os resultados aparecem em `resultados/` nesta pasta. Para outra GPU,
altere `CUDA_ARCH` em `docker-compose.yml` (o padrão é `sm_61`, da GTX 1050 Ti).


## Problemas comuns observados

| Erros observados | Soluções |

| **nvcc: command not found / não é reconhecido** | CUDA fora do `PATH` (reabra o terminal) ou toolkit não instalado |
| **nvcc fatal: Unsupported gpu architecture 'compute_61** | CUDA 13 instalado; use CUDA 12.x |
| **nvcc error : 'cudafe++' died with status 0xC0000409** (Windows) | MSVC novo demais (toolset 14.5x); instale o v143 14.36-17.6 |
| **Cannot find compiler 'cl.exe'** (Windows) | Compile pelo `compilar_windows.bat`, não direto pelo `mingw32-make` |
| **Erro de DLL ao abrir `tccublas.exe`** | A pasta `bin` do CUDA não está no `PATH` (`cublas64_12.dll`) |
| **Erro na operacao ... dimensoes incompativeis** | Soma/subtração devem ter dimensões iguais; multiplicação: o número de colunas de A devem ser igual ao de linhas de B |



## Formato dos arquivos TXT que são lidos pela biblioteca

```
<linhas> <colunas>
a00 a01 ... a0(c-1)
a10 a11 ... a1(c-1)
...
```
Valores float32 em ordem row-major, separados por espaço. O gerador grava 9
dígitos significativos, o suficiente para recuperar o float32 exatamente. Leitura
em C:
```c
FILE *f = fopen("matriz_a.txt", "r");
int linhas, colunas;
fscanf(f, "%d %d", &linhas, &colunas);
float *m = (float *) malloc(sizeof(float) * linhas * colunas);
for (long i = 0; i < (long) linhas * colunas; i++)
    fscanf(f, "%f", &m[i]);
fclose(f);
```


## Binding Python

`tccublas.py` chama a biblioteca compilada (`libTCCuBLAS.dll` / `.so`, na mesma
pasta) via `ctypes`. Ele recebe matrizes NumPy (convertidas para float32
C-contíguo), executa a operação na GPU e devolve o resultado como array NumPy
float32 junto aos tempos medidos dentro da biblioteca.



```python
import numpy as np
from TCCuBLAS import tccublas      # ou: import tccublas, dentro desta pasta

a = np.random.rand(100, 50).astype(np.float32)
b = np.random.rand(50, 20).astype(np.float32)

c, tempos = tccublas.multiplicar(a, b)     # também: somar(a, b), subtrair(a, b)
print(c.shape, tempos["calculo_ms"])       # (100, 20) e o tempo de cálculo em ms

r, tempos = tccublas.multiplicar_cadeia([a, b])   # A1 @ A2 @ ... @ An
m = tccublas.ler_matriz("matriz_a.txt")           # lê um TXT pela própria biblioteca
```

- `tempos` é um dicionário com `h2d_ms`, `calculo_ms`, `d2h_ms` e `total_ms`.
- Erros da biblioteca (dimensões incompatíveis, arquivo inválido etc.) viram
  `tccublas.TCCudaErro`, com o código em `.codigo`.
- `multiplicar_cadeia` multiplica da **esquerda para a direita**; a cada passo o
  resultado parcial volta ao host e é enviado de novo à GPU. A ordem ótima de
  parentização e a cadeia residente na GPU são a próxima etapa.
- No Windows, o binding registra sozinho a pasta `bin` do CUDA (`CUDA_PATH`) para
  achar a `cublas64_*.dll`.



**Testes do binding** (com a biblioteca já compilada, como mostrado nas sessões acima), compara a TCCuBLAS com o
NumPy em soma, subtração, multiplicação, cadeias, conversão de tipo, erros e
leitura de arquivo:
```powershell
python teste_binding.py
```


