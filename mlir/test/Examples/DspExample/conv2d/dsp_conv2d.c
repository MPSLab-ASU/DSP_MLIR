#include <stdio.h>
#include <stdlib.h>

void freeArr(int size, int** arr) {
    for(int i=0; i<size; ++i) {
        free(arr[i]);
    }
}

void conv2d(int INPUTSIZE, int KERNELSIZE, int** a, int **b, int **c) {
    int outputsize = INPUTSIZE-KERNELSIZE+1;
    for(int i=0; i<outputsize; ++i) {
        for(int j=0; j<outputsize; ++j) {
            c[i][j] = 0;
            for(int p=0; p<KERNELSIZE; ++p) {
                for(int q=0; q<KERNELSIZE; ++q) {
                    if(i+p >= INPUTSIZE || j+q >= INPUTSIZE) continue;
                    c[i][j] += a[i+p][j+q] * b[p][q];

                }
            }
        }
    }
}

int main() {

    const char* filename = "input.txt";
    FILE *file = fopen(filename, "r");

    if(file == NULL) {
        printf("error opening file");
        exit(1);
    }
    int* inputrows, *inputcols, *kernelrows, *kernelcols;
    int rows=0, cols=0;
    fscanf(file, "%d %d", &rows, &cols);

    inputrows = &rows;
    inputcols = &cols;

    printf("input size %d, %d\n", *inputrows, *inputcols);
    int** a = (int**) malloc((*inputrows)*sizeof(int*));
    for(int i=0; i<(*inputrows); ++i) {
        a[i] = (int*) malloc((*inputcols)*sizeof(int));
    }

    for(int i=0; i<(*inputrows); ++i) {
        for(int j=0; j<(*inputcols); ++j) {
            fscanf(file, "%d ", &a[i][j]);
        }
    }

    int krows=0, kcols=0;
    fscanf(file, "%d %d", &krows, &kcols);

    kernelrows = &krows;
    kernelcols = &kcols;

    printf("kernel size %d, %d\n", *kernelrows, *kernelcols);
    int** b = (int**) malloc((*kernelrows)*sizeof(int*));
    for(int i=0; i<(*kernelrows); ++i) {
        b[i] = (int*) malloc((*kernelcols)*sizeof(int));
    }

    for(int i=0; i<(*kernelrows); ++i) {
        for(int j=0; j<(*kernelcols); ++j) {
            fscanf(file, "%d", &b[i][j]);
        }
    }

    fclose(file);


    int outputrows = (*inputrows) - (*kernelrows) +1;
    int outputcols = (*inputcols) - (*kernelcols) +1;
    int** c = (int**) malloc((outputrows)*sizeof(int*));
    for(int i=0; i<outputrows; ++i){
        c[i] = (int*) malloc(outputcols*sizeof(int));
    }
    printf("output size %d, %d\n", outputrows, outputcols);

    conv2d((*inputrows), (*kernelrows), a, b, c);

    for(int i=0; i<outputrows; ++i) {
        for(int j=0; j<outputcols; ++j) {
            printf("%d ", c[i][j]);
        }

        printf("\n");
    }

    freeArr((*inputrows), a);
    freeArr((*kernelrows), b);
    freeArr((outputrows), c);
    return 0;
}
