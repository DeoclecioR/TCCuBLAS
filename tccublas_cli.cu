
#include "TCCuBLAS.h"

#include <stdio.h>
#include <string.h>

static void uso(const char* prog) {
    fprintf(stderr,
            "Uso: %s <soma|subtracao|multiplicacao> matriz_a.txt matriz_b.txt [--sem-matriz]\n",
            prog);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        uso(argv[0]);
        return 1;
    }

    TCCudaOperacao op;
    if (strcmp(argv[1], "soma") == 0) op = TCCUDA_SOMAR;
    else if (strcmp(argv[1], "subtracao") == 0) op = TCCUDA_SUBTRAIR;
    else if (strcmp(argv[1], "multiplicacao") == 0) op = TCCUDA_MULTIPLICAR;
    else {
        uso(argv[0]);
        return 1;
    }

    int imprimir = 1;
    if (argc >= 5 && strcmp(argv[4], "--sem-matriz") == 0) imprimir = 0;

    return TCCuda_executar(op, argv[2], argv[3], imprimir);
}
