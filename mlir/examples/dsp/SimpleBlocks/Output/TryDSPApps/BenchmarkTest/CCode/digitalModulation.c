#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>

#define INPUT_LENGTH 100000000

void generate_input(int *data) {
    for(int i=0; i<INPUT_LENGTH; ++i) {
        data[i] = rand() %2;
    }
}

void qam_modulate(complex double *symbols, int *data) {
    for(int i=0; i<INPUT_LENGTH; i+=2) {
        int bit1 = data[i];
        int bit2 = data[i+1];

        if(bit1 == 0 && bit2==0)
            symbols[i/2] = -1 - 1*I;
        else if(bit1 == 0&& bit2==1)
            symbols[i/2] = -1 + 1*I;
        else if(bit1==1 && bit2==0)
            symbols[i/2] = 1-1*I;
        else if (bit1==1 && bit2==1)
            symbols[i/2] = 1+1*I;
    }
}

void qam_demodulate(int *bits, complex double *symbols) {
    for (int i = 0; i < INPUT_LENGTH/2; i++) {
        complex double symbol = symbols[i];

        if (symbol == -1 - 1*I) {
            bits[2 * i] = 0;
            bits[2 * i + 1] = 0;
        } else if (symbol == -1 + 1*I) {
            bits[2 * i] = 0;
            bits[2 * i + 1] = 1;
        } else if (symbol == 1 - 1*I) {
            bits[2 * i] = 1;
            bits[2 * i + 1] = 0;
        } else if (symbol == 1 + 1*I) {
            bits[2 * i] = 1;
            bits[2 * i + 1] = 1;
        }
    }
}

int main() {
    srand(time(NULL)); // Seed random number generator
    
    int* data;
    data = (int*)malloc(sizeof(int) * INPUT_LENGTH);
    complex double* symbols;
    symbols = (complex double*)malloc(sizeof(complex double) * INPUT_LENGTH / 2);
    
    // Generate random input data
    generate_input(data);
    
    // Perform QAM modulation
    qam_modulate(symbols, data);
    
    // Perform QAM demodulation
    int* bits; 
    bits = (int*)malloc(sizeof(int) * INPUT_LENGTH);
    qam_demodulate(bits, symbols);

    // Print results
    for (int i = 0; i < INPUT_LENGTH; i++) {
        printf("%d ", bits[i]);
    }
    printf("\n");
    
    
    return 0;
}