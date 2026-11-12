
// TCCuBLAS.cu

// Implementacao da biblioteca TCCuBLAS (ver TCCuBLAS.h).

// A primeira chamada de cuBLAS carrega kernels sob demanda;
// por isso se faz uma chamada de "aquecimento" antes de medir.

// Compilar: ver Makefile (libTCCuBLAS.so e executavel tccublas).


#include "TCCuBLAS.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdlib.h>
#include <string.h>

const char* TCCuda_mensagem_erro(int codigo) {
    switch (codigo) {
        case TCCUDA_DUZENTOS_OK:    return "sem erro";
        case TCCUDA_ERRO_ARGUMENTO: return "argumento invalido";
        case TCCUDA_ERRO_ARQUIVO:   return "falha ao ler o arquivo (inexistente ou formato invalido)";
        case TCCUDA_ERRO_DIMENSOES: return "dimensoes incompativeis com a operacao";
        case TCCUDA_ERRO_MEMORIA:   return "memoria insuficiente no host";
        case TCCUDA_ERRO_CUDA:      return "erro CUDA (memoria da GPU ou transferencia)";
        case TCCUDA_ERRO_CUBLAS:    return "erro cuBLAS";
        default:             return "erro desconhecido";
    }
}



// ---------------------------------------------------------------------------
// Matrizes no host(CPU)
// ---------------------------------------------------------------------------
static TCCudaMatriz* alocar_matriz(int linhas, int colunas) {
    TCCudaMatriz* m = (TCCudaMatriz*) malloc(sizeof(TCCudaMatriz));
    if (m == NULL) return NULL;
    m->linhas = linhas;
    m->colunas = colunas;
    m->dados = (float*) malloc(sizeof(float) * (size_t) linhas * colunas);
    if (m->dados == NULL) {
        free(m);
        return NULL;
    }
    return m;
}

void TCCuda_liberar_matriz(TCCudaMatriz* m) {
    if (m == NULL) return;
    free(m->dados);
    free(m);
}

TCCudaMatriz* TCCuda_ler_matriz(const char* caminho, int* erro) {
    if (erro != NULL) *erro = TCCUDA_DUZENTOS_OK;
    if (caminho == NULL) {
        if (erro != NULL) *erro = TCCUDA_ERRO_ARGUMENTO;
        return NULL;
    }
    FILE* f = fopen(caminho, "r");
    if (f == NULL) {
        if (erro != NULL) *erro = TCCUDA_ERRO_ARQUIVO;
        return NULL;
    }

    int linhas = 0, colunas = 0;
    if (fscanf(f, "%d %d", &linhas, &colunas) != 2 || linhas < 1 || colunas < 1) {
        fclose(f);
        if (erro != NULL) *erro = TCCUDA_ERRO_ARQUIVO;
        return NULL;
    }

    TCCudaMatriz* m = alocar_matriz(linhas, colunas);
    if (m == NULL) {
        fclose(f);
        if (erro != NULL) *erro = TCCUDA_ERRO_MEMORIA;
        return NULL;
    }

    size_t total = (size_t) linhas * colunas;
    for (size_t i = 0; i < total; i++) {
        if (fscanf(f, "%f", &m->dados[i]) != 1) {
            fclose(f);
            TCCuda_liberar_matriz(m);
            if (erro != NULL) *erro = TCCUDA_ERRO_ARQUIVO;
            return NULL;
        }
    }
    fclose(f);
    return m;
}

// ---------------------------------------------------------------------------
// Execucao na GPU
// ---------------------------------------------------------------------------
// Operacao interna compartilhada pelas tres funcoes publicas.
static int executar_gpu(TCCudaOperacao op, const TCCudaMatriz* a, const TCCudaMatriz* b,
                        TCCudaMatriz** resultado, TCCudaTempos* tempos) {
    if (a == NULL || b == NULL || resultado == NULL ||
        a->dados == NULL || b->dados == NULL) return TCCUDA_ERRO_ARGUMENTO;
    *resultado = NULL;

    int m, n;  // dimensoes do resultado (m linhas x n colunas)
    if (op == TCCUDA_MULTIPLICAR) {
        if (a->colunas != b->linhas) return TCCUDA_ERRO_DIMENSOES;
        m = a->linhas;
        n = b->colunas;
    } else {
        if (a->linhas != b->linhas || a->colunas != b->colunas) return TCCUDA_ERRO_DIMENSOES;
        m = a->linhas;
        n = a->colunas;
    }

    size_t bytes_a = sizeof(float) * (size_t) a->linhas * a->colunas;
    size_t bytes_b = sizeof(float) * (size_t) b->linhas * b->colunas;
    size_t bytes_c = sizeof(float) * (size_t) m * n;

    TCCudaMatriz* c = alocar_matriz(m, n);
    if (c == NULL) return TCCUDA_ERRO_MEMORIA;

    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    cublasHandle_t handle = NULL;
    cudaEvent_t ev[6];
    for (int i = 0; i < 6; i++) ev[i] = NULL;
    float ms_h2d = 0.f, ms_calc = 0.f, ms_d2h = 0.f;
    int rc = TCCUDA_DUZENTOS_OK;

    // Cria o contexto CUDA, o handle e as alocacoes.
    if (cudaFree(0) != cudaSuccess) rc = TCCUDA_ERRO_CUDA;
    if (rc == TCCUDA_DUZENTOS_OK && cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        handle = NULL;
        rc = TCCUDA_ERRO_CUBLAS;
    }
    if (rc == TCCUDA_DUZENTOS_OK) cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);  // float32 puro
    if (rc == TCCUDA_DUZENTOS_OK && (cudaMalloc((void**) &d_a, bytes_a) != cudaSuccess ||
                         cudaMalloc((void**) &d_b, bytes_b) != cudaSuccess ||
                         cudaMalloc((void**) &d_c, bytes_c) != cudaSuccess))
        rc = TCCUDA_ERRO_CUDA;
    for (int i = 0; i < 6 && rc == TCCUDA_DUZENTOS_OK; i++)
        if (cudaEventCreate(&ev[i]) != cudaSuccess) rc = TCCUDA_ERRO_CUDA;

    const float um = 1.0f, zero = 0.0f, menos_um = -1.0f;

    // Chamada de aquecimento
    if (rc == TCCUDA_DUZENTOS_OK) {
        cudaMemset(d_a, 0, bytes_a);
        cudaMemset(d_b, 0, bytes_b);
        cublasStatus_t s;
        if (op == TCCUDA_MULTIPLICAR)
            s = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, a->colunas,
                            &um, d_b, n, d_a, a->colunas, &zero, d_c, n);
        else
            s = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m,
                            &um, d_a, n, &um, d_b, n, d_c, n);
        if (s != CUBLAS_STATUS_SUCCESS) rc = TCCUDA_ERRO_CUBLAS;
        if (rc == TCCUDA_DUZENTOS_OK && cudaDeviceSynchronize() != cudaSuccess) rc = TCCUDA_ERRO_CUDA;
    }

    // (1/3) host -> device
    if (rc == TCCUDA_DUZENTOS_OK) {
        cudaEventRecord(ev[0], 0);
        if (cudaMemcpy(d_a, a->dados, bytes_a, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_b, b->dados, bytes_b, cudaMemcpyHostToDevice) != cudaSuccess)
            rc = TCCUDA_ERRO_CUDA;
        cudaEventRecord(ev[1], 0);
    }

    // (2/3) calculo
    if (rc == TCCUDA_DUZENTOS_OK) {
        cublasStatus_t s;
        cudaEventRecord(ev[2], 0);
        if (op == TCCUDA_MULTIPLICAR) {
            // C(m x n) = A(m x k) * B(k x n)  <=>  C^T = B^T * A^T
            s = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, a->colunas,
                            &um, d_b, n, d_a, a->colunas, &zero, d_c, n);
        } else {
            const float* beta = (op == TCCUDA_SOMAR) ? &um : &menos_um;
            s = cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m,
                            &um, d_a, n, beta, d_b, n, d_c, n);
        }
        cudaEventRecord(ev[3], 0);
        if (s != CUBLAS_STATUS_SUCCESS) rc = TCCUDA_ERRO_CUBLAS;
    }

    // (3/3) device -> host
    if (rc == TCCUDA_DUZENTOS_OK) {
        cudaEventRecord(ev[4], 0);
        if (cudaMemcpy(c->dados, d_c, bytes_c, cudaMemcpyDeviceToHost) != cudaSuccess)
            rc = TCCUDA_ERRO_CUDA;
        cudaEventRecord(ev[5], 0);
    }

    if (rc == TCCUDA_DUZENTOS_OK) {
        if (cudaEventSynchronize(ev[5]) != cudaSuccess) {
            rc = TCCUDA_ERRO_CUDA;
        } else {
            cudaEventElapsedTime(&ms_h2d, ev[0], ev[1]);
            cudaEventElapsedTime(&ms_calc, ev[2], ev[3]);
            cudaEventElapsedTime(&ms_d2h, ev[4], ev[5]);
        }
    }

    // Limpeza da memória.
    for (int i = 0; i < 6; i++)
        if (ev[i] != NULL) cudaEventDestroy(ev[i]);
    if (d_a != NULL) cudaFree(d_a);
    if (d_b != NULL) cudaFree(d_b);
    if (d_c != NULL) cudaFree(d_c);
    if (handle != NULL) cublasDestroy(handle);

    if (rc != TCCUDA_DUZENTOS_OK) {
        TCCuda_liberar_matriz(c);
        return rc;
    }

    if (tempos != NULL) {
        tempos->h2d_ms = ms_h2d;
        tempos->calculo_ms = ms_calc;
        tempos->d2h_ms = ms_d2h;
        tempos->total_ms = (double) ms_h2d + ms_calc + ms_d2h;
    }
    *resultado = c;
    return TCCUDA_DUZENTOS_OK;
}

int TCCuda_somar(const TCCudaMatriz* a, const TCCudaMatriz* b,
              TCCudaMatriz** resultado, TCCudaTempos* tempos) {
    return executar_gpu(TCCUDA_SOMAR, a, b, resultado, tempos);
}

int TCCuda_subtrair(const TCCudaMatriz* a, const TCCudaMatriz* b,
                 TCCudaMatriz** resultado, TCCudaTempos* tempos) {
    return executar_gpu(TCCUDA_SUBTRAIR, a, b, resultado, tempos);
}

int TCCuda_multiplicar(const TCCudaMatriz* a, const TCCudaMatriz* b,
                    TCCudaMatriz** resultado, TCCudaTempos* tempos) {
    return executar_gpu(TCCUDA_MULTIPLICAR, a, b, resultado, tempos);
}

// ---------------------------------------------------------------------------
// Saida
// ---------------------------------------------------------------------------
void TCCuda_imprimir_resultado(const TCCudaMatriz* r, const TCCudaTempos* t,
                            int imprimir_matriz, FILE* saida) {
    if (saida == NULL) saida = stdout;
    if (r == NULL) return;

    fprintf(saida, "Dimensoes da matriz resultante: %d x %d\n", r->linhas, r->colunas);

    if (imprimir_matriz) {
        fprintf(saida, "Matriz resultante:\n");
        for (int i = 0; i < r->linhas; i++) {
            for (int j = 0; j < r->colunas; j++)
                fprintf(saida, "%.6f ", r->dados[(size_t) i * r->colunas + j]);
            fprintf(saida, "\n");
        }
    }

    if (t != NULL) {
        fprintf(saida, "Tempo decorrido (GPU):\n");
        fprintf(saida, "  transferencia host->device : %.4f ms\n", t->h2d_ms);
        fprintf(saida, "  calculo                    : %.4f ms\n", t->calculo_ms);
        fprintf(saida, "  transferencia device->host : %.4f ms\n", t->d2h_ms);
        fprintf(saida, "  total                      : %.4f ms\n", t->total_ms);
    }
}

int TCCuda_executar(TCCudaOperacao op, const char* arquivo_a, const char* arquivo_b,
                 int imprimir_matriz) {
    int erro = TCCUDA_DUZENTOS_OK;
    TCCudaMatriz* a = TCCuda_ler_matriz(arquivo_a, &erro);
    if (a == NULL) {
        fprintf(stderr, "Erro ao ler '%s': %s\n", arquivo_a ? arquivo_a : "(null)",
                TCCuda_mensagem_erro(erro));
        return erro;
    }
    TCCudaMatriz* b = TCCuda_ler_matriz(arquivo_b, &erro);
    if (b == NULL) {
        fprintf(stderr, "Erro ao ler '%s': %s\n", arquivo_b ? arquivo_b : "(null)",
                TCCuda_mensagem_erro(erro));
        TCCuda_liberar_matriz(a);
        return erro;
    }

    TCCudaMatriz* r = NULL;
    TCCudaTempos t;
    memset(&t, 0, sizeof(t));
    int rc;
    switch (op) {
        case TCCUDA_SOMAR:       rc = TCCuda_somar(a, b, &r, &t); break;
        case TCCUDA_SUBTRAIR:    rc = TCCuda_subtrair(a, b, &r, &t); break;
        case TCCUDA_MULTIPLICAR: rc = TCCuda_multiplicar(a, b, &r, &t); break;
        default:              rc = TCCUDA_ERRO_ARGUMENTO; break;
    }

    if (rc != TCCUDA_DUZENTOS_OK) {
        fprintf(stderr, "Erro na operacao (A %dx%d, B %dx%d): %s\n",
                a->linhas, a->colunas, b->linhas, b->colunas, TCCuda_mensagem_erro(rc));
    } else {
        TCCuda_imprimir_resultado(r, &t, imprimir_matriz, stdout);
    }

    TCCuda_liberar_matriz(r);
    TCCuda_liberar_matriz(a);
    TCCuda_liberar_matriz(b);
    return rc;
}
