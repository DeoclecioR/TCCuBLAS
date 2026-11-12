# TCCuBLAS

Biblioteca em C/CUDA que realiza operações entre matrizes densas, com
float32 na GPU** usando o cuBLAS. Ela lê as matrizes de arquivos TXT e imprime as
dimensões do resultado, a matriz e o tempo decorrido.


- GPU de referência: NVIDIA GeForce GTX 1050 Ti, 4 GB (Pascal, `sm_61`).
- Funciona como biblioteca compartilhada (`libTCCuBLAS.dll` / `.so`) e como
  executável de linha de comando (`tccublas`).

## Estrutura

| Arquivo | Função |
|---|---|
| `TCCuBLAS.h` | API pública (tipos, constantes e funções) |
| `TCCuBLAS.cu` | Implementação |
| `tccublas_cli.cu` | Programa de linha de comando para executar a biblioteca
| `Makefile` | Compila a biblioteca e o executável (Windows e Linux) |
| `compilar_windows.bat` | No Windows, localiza o MSVC e chama o `Makefile` |
| `auxiliares/gerar_matrizes_txt.py` | Gera duas matrizes float32 em TXT |

## Softwares necessários

Uma **GPU NVIDIA** com driver instalado (`nvidia-smi` deve funcionar). Versões
usadas e validadas:

| Software | Windows | 
|---|---|---|
| Driver NVIDIA | 582.78 | 
| CUDA Toolkit (`nvcc`, cuBLAS) | 12.6 | 
| Compilador C/C++ | MSVC v143, toolset 14.36 (VS 2022 17.6) | 
| Make | GNU Make 4.4.1 (`mingw32-make`) | 
| Python + NumPy (para gerar as matrizes) | 3.12 + NumPy 1.26 | 


- **Não funciona com CUDA 13 em diante.**
- Ele removeu o suporte à arquitetura Pascal (`sm_61`), e o `nvcc` não compila para a GTX 1050 Ti.

- **No Windows, o `nvcc` só compila com o MSVC** (o MinGW não serve como compilador do CUDA).
Um MSVC novo demais (toolset 14.5x, Visual Studio 2026) derruba o `nvcc` (`cudafe++ died with status 0xC0000409`). 
Instale o componente **MSVC v143 – VS 2022 C++ x64/x86 build tools (v14.36-17.6)** no Visual Studio
Installer, em *Componentes individuais*, junto com a carga *Desenvolvimento para desktop com C++*.
  
 O `compilar_windows.bat` usa automaticamente o toolset `14.4*` ou `14.3*` que encontrar.
- No Windows, o `mingw32-make` só executa o `Makefile`; quem compila é o `nvcc` com o MSVC.

## Compilar

**Windows** (PowerShell, na pasta do projeto):
```powershell
.\compilar_windows.bat
```

Gera `libTCCuBLAS.dll` (ou `.so`) e o executável `tccublas.exe`. 
 Para limpar: `make clean` ou `.\compilar_windows.bat clean`.

## Usar o executável

1. Gere duas matrizes (os quatro inteiros são linhas e colunas de A e de B):
   ```bash
   python auxiliares/gerar_matrizes_txt.py 1000 500 500 2000 --saida-a matriz_a.txt --saida-b matriz_b.txt
   ```
2. Execute a operação:
   ```bash
   ./tccublas multiplicacao matriz_a.txt matriz_b.txt       # Windows: .\tccublas.exe
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

## Formato dos arquivos TXT

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



## Problemas comuns

| Sintoma | Causa provável |
|---|---|
| `nvcc: command not found` / `não é reconhecido` | CUDA fora do `PATH` (reabra o terminal) ou toolkit não instalado |
| `nvcc fatal: Unsupported gpu architecture 'compute_61'` | CUDA 13 instalado; use CUDA 12.x |
| `nvcc error : 'cudafe++' died with status 0xC0000409` (Windows) | MSVC novo demais (toolset 14.5x); instale o v143 14.36-17.6 |
| `Cannot find compiler 'cl.exe'` (Windows) | Compile pelo `compilar_windows.bat`, não direto pelo `mingw32-make` |
| Erro de DLL ao abrir `tccublas.exe` | A pasta `bin` do CUDA não está no `PATH` (`cublas64_12.dll`) |
| `Erro na operacao ... dimensoes incompativeis` | Soma/subtração: dimensões iguais; multiplicação: colunas de A = linhas de B |
