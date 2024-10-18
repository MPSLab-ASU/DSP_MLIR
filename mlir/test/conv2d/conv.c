#include <stdio.h>
#include <stdlib.h>

#define INPUTSIZE 4
#define KERNELSIZE 3

double example_in[INPUTSIZE][INPUTSIZE] = {
    {1,2,3,4},
    {2,3,4,6},
    {4,3,2,1},
    {6,8,4,7}
};

double example_kernel[KERNELSIZE][KERNELSIZE] = {
    {1,0,-1},
    {1,0,-1},
    {1,0,-1}
};

void conv2d(double **input, double** kernel, double** output) {
    int delta = KERNELSIZE / 2;

    for(int x=0; x<INPUTSIZE; ++x) {
        for(int y=0; y<INPUTSIZE; ++y) {
            float sum=0;

            for(int kx=-1*delta; kx<=delta; ++kx) {
                for(int ky=-1*delta; ky<=delta; ++ky) {
                    int imgX = x+kx, imgY = y+ky;
                    if(imgX>=0 && imgY>=0 && imgX<INPUTSIZE && imgY<INPUTSIZE) {
                        float imgVal = input[imgX][imgY];
                        float kerVal = kernel[kx+delta][ky+delta];
                        sum += imgVal * kerVal;
                    }
                }
            }

            output[x][y] = sum;
        }
    }
};

int main() {
    double ** input = (double**) malloc( INPUTSIZE * sizeof(double*) );
    double ** kernel = (double**) malloc( KERNELSIZE * sizeof(double*) );
    double ** output = (double**) malloc( INPUTSIZE * sizeof(double*) );

    for(int i=0; i<INPUTSIZE; ++i) {
        input[i] = (double*) malloc( INPUTSIZE * sizeof(double) );
        output[i] = (double*) malloc( INPUTSIZE * sizeof(double) );
    }

    for(int i=0; i<KERNELSIZE; ++i) {
        kernel[i] = (double*) malloc( KERNELSIZE * sizeof(double) );
    }

    for(int x=0; x<INPUTSIZE; ++x) {
        for(int y=0; y<INPUTSIZE; ++y) {
            input[x][y] = example_in[x][y];
        }
    }
    
    for(int x=0; x<KERNELSIZE; ++x) {
        for(int y=0; y<KERNELSIZE; ++y) {
            kernel[x][y] = example_kernel[x][y];
        }
    }

    // conv
    conv2d(input, kernel, output);

    printf("Output:\n");
    for(int x=0; x<INPUTSIZE; ++x) {
        for(int y=0; y<INPUTSIZE; ++y) {
            printf("%f ", output[x][y]);
        }
        printf("\n");
    }

    for(int x=0; x<INPUTSIZE; ++x) {
        free(input[x]);
        free(output[x]);
    }

    free(input);
    free(output);

    for(int x=0; x<KERNELSIZE; ++x) {
        free(kernel[x]);
    }

    free(kernel);

    return 0;
}
