/*
Formato do TXT lido (gerado por gerar_matrizes_txt.py):
     <linhas> <colunas>
     valores row-major separados por espaco
*/


#ifndef TCCUBLAS_H
#define TCCUBLAS_H

#include <stdio.h>


#if defined(_WIN32)
#define TCCUDA_API __declspec(dllexport)
#else
#define TCCUDA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    int linhas;
    int colunas;
    float* dados;   
} TCCudaMatriz;

// Tempos medidos com cudaEvent em milisegundos(ms).
typedef struct {
    double h2d_ms;      // host to(2) device
    double calculo_ms;  // operacao feita na GPU
    double d2h_ms;      // transferencia device to host (resultado)
    double total_ms;    // soma dos tempos acima
} TCCudaTempos;

typedef enum {
    TCCUDA_SOMAR = 0,
    TCCUDA_SUBTRAIR = 1,
    TCCUDA_MULTIPLICAR = 2
} TCCudaOperacao;

// Codigos de retorno.
enum {
    TCCUDA_DUZENTOS_OK = 0,        //Sem erro 
    TCCUDA_ERRO_ARGUMENTO = 1,     //O argumento inserido é inválido
    TCCUDA_ERRO_ARQUIVO = 2,       // nao abriu ou formato invalido
    TCCUDA_ERRO_DIMENSOES = 3,     // dimensoes incompativeis com a operacao
    TCCUDA_ERRO_MEMORIA = 4,       // malloc no host
    TCCUDA_ERRO_CUDA = 5,          // erro com driver da placa
    TCCUDA_ERRO_CUBLAS = 6         // erro no framework CUDA
};


TCCUDA_API const char* TCCuda_mensagem_erro(int codigo);



TCCUDA_API TCCudaMatriz* TCCuda_ler_matriz(const char* caminho, int* erro);
TCCUDA_API void TCCuda_liberar_matriz(TCCudaMatriz* m);



TCCUDA_API int TCCuda_somar(const TCCudaMatriz* a, const TCCudaMatriz* b,
              TCCudaMatriz** resultado, TCCudaTempos* tempos);
TCCUDA_API int TCCuda_subtrair(const TCCudaMatriz* a, const TCCudaMatriz* b,
                 TCCudaMatriz** resultado, TCCudaTempos* tempos);
TCCUDA_API int TCCuda_multiplicar(const TCCudaMatriz* a, const TCCudaMatriz* b,
                    TCCudaMatriz** resultado, TCCudaTempos* tempos);

TCCUDA_API void TCCuda_imprimir_resultado(const TCCudaMatriz* r, const TCCudaTempos* t,
                            int imprimir_matriz, FILE* saida);



TCCUDA_API int TCCuda_executar(TCCudaOperacao op, const char* arquivo_a, const char* arquivo_b,
                 int imprimir_matriz);

#ifdef __cplusplus
}
#endif

#endif // TCCUBLAS_H
