#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#define PI 3.1415926
#define INPUT_LENGTH 10000

// Function prototypes
double* getrangeofvector(double first, int64_t N, double step);
double* beamForm(int antennas, double frequency, double* time, double* weights, int timeDim);
double* abs_array(double* arr, int size);
double* power_profile(double* arr, int size);
double* lowPassFIRFilter(double wc, int N);
double* highPassFIRFilter(double wc, int N);
double* hamming(int N);
double* multiply_arrays(const double* arr1, const double* arr2, int size);
double* subtract_arrays(const double* arr1, const double* arr2, int size);
double* FirFilterResponse(const double *input, int inputLen, const double *filter, int filterLen);

int main() {
    // Parameters
    int antennas = 4;
    double input_fc = 5;
    int N = 101;
    int input_length = INPUT_LENGTH;
    double fc1 = 1000;
    double fc2 = 7500;
    double Fs = 8000;

    double* input = getrangeofvector(0, input_length, 0.000125);
    double* weights = getrangeofvector(-90, 180, 1);
    double* signal = beamForm(antennas, input_fc, input, weights, input_length);
    double* b1 = abs_array(signal, input_length);
    double* power = power_profile(b1, input_length);
    double wc1 = 2 * PI * fc1 / Fs;
    double* filter1 = lowPassFIRFilter(wc1, N);
    double* filter_hamming_1 = multiply_arrays(filter1, hamming(N), N);
    double wc2 = 2 * PI * fc2 / Fs;
    double* filter2 = highPassFIRFilter(wc2, N);
    double* filter_hamming_2 = multiply_arrays(filter2, hamming(N), N);
    double* bpf = subtract_arrays(filter_hamming_2, filter_hamming_1, N);
    double* firFilterResponse = FirFilterResponse(power, input_length, bpf, N);
    double final = firFilterResponse[10099];
    printf("final: %f\n", final);

    // for (int i = 0; i < (input_length + N - 1); ++i) {
    //     printf("firFilterResponse: %f\n", firFilterResponse[i]);
    // }
    
    // Free allocated memory
    free(input);
    free(weights);
    free(signal);
    free(b1);
    free(power);
    free(filter1);
    free(filter2);
    free(filter_hamming_1);
    free(filter_hamming_2);
    free(bpf);
    free(firFilterResponse);
    return 0;
}

double* getrangeofvector(double first, int64_t N, double step) {
    double* result = (double*)malloc(N * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    // Initialize the first element
    result[0] = first;
    
    // Calculate the rest of the elements
    for (int64_t i = 1; i < N; ++i) {
        result[i] = result[i-1] + step;
    }
    
    return result;
}

double* beamForm(int antennas, double frequency, double* time, double* weights, int timeDim) {
    // Allocate space for output
    double* output = (double*)malloc(timeDim * sizeof(double));
    if (output == NULL) {
        fprintf(stderr, "Memory allocation failed for output\n");
        exit(1);
    }

    // Allocate space for internal generated signals
    double** signal = (double**)malloc(antennas * sizeof(double*));
    if (signal == NULL) {
        fprintf(stderr, "Memory allocation failed for signal\n");
        free(output);
        exit(1);
    }
    
    for (int i = 0; i < antennas; i++) {
        signal[i] = (double*)malloc(timeDim * sizeof(double));
        if (signal[i] == NULL) {
            fprintf(stderr, "Memory allocation failed for signal[%d]\n", i);
            for (int j = 0; j < i; j++) {
                free(signal[j]);
            }
            free(signal);
            free(output);
            exit(1);
        }
    }

    // Generate input signals
    double phase_var = 2 * PI * frequency;
    for (int i = 0; i < antennas; i++) {
        double iter_args = (i * PI) / 4.0;
        for (int j = 0; j < timeDim; j++) {
            double sin_body = time[j] * phase_var + iter_args;
            signal[i][j] = sin(sin_body);
        }
    }

    // Beam forming
    for (int i = 0; i < timeDim; i++) {
        double sum = 0.0;
        for (int j = 0; j < antennas; j++) {
            sum += signal[j][i] * weights[j];
        }
        output[i] = sum;
    }

    // Free allocated memory for signal
    for (int i = 0; i < antennas; i++) {
        free(signal[i]);
    }
    
    free(signal);

    return output;
}

// Function to calculate absolute values of an array
double* abs_array(double* arr, int size) {
    double* result = (double*)malloc(size * sizeof(double));
    
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        result[i] = fabs(arr[i]);
    }

    return result;
}

// Function to calculate power profile (element-wise square)
double* power_profile(double* arr, int size) {
    double* result = (double*)malloc(size * sizeof(double));
    
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        result[i] = arr[i] * arr[i];
    }

    return result;
}

double* lowPassFIRFilter(double wc, int N) {
    double* output = (double*)malloc(N * sizeof(double));
    if (output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int midIndex = (N - 1) / 2;
    double wcByPi = wc / PI;

    // Handle middle point
    output[midIndex] = wcByPi;

    // First loop: 0 <= i <= (N-1)/2 - 1
    for (int i = 0; i < midIndex; i++) {
        double iMinusMid = i - midIndex;
        double sinArg = wc * iMinusMid;
        double sinValue = sin(sinArg);
        output[i] = sinValue / (PI * iMinusMid);
    }

    // Second loop: (N-1)/2 + 1 <= i < N
    for (int i = midIndex + 1; i < N; i++) {
        double iMinusMid = i - midIndex;
        double sinArg = wc * iMinusMid;
        double sinValue = sin(sinArg);
        output[i] = sinValue / (PI * iMinusMid);
    }

    return output;
}


double* highPassFIRFilter(double wc, int N) {
    double* output = (double*)malloc(N * sizeof(double));
    if (output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    int midIndex = (N - 1) / 2;
    double wcByPi = wc / PI;

    // Handle middle point
    output[midIndex] = 1.0 - wcByPi;

    // First loop: 0 <= i <= (N-1)/2 - 1
    for (int i = 0; i < midIndex; i++) {
        double iMinusMid = i - midIndex;
        double sinArg = wc * iMinusMid;
        double sinValue = sin(sinArg);
        output[i] = -1.0 * sinValue / (PI * iMinusMid);
    }

    // Second loop: (N-1)/2 + 1 <= i < N
    for (int i = midIndex + 1; i < N; i++) {
        double iMinusMid = i - midIndex;
        double sinArg = wc * iMinusMid;
        double sinValue = sin(sinArg);
        output[i] = -1.0 * sinValue / (PI * iMinusMid);
    }

    return output;
}

double* hamming(int N) {
    double* window = (double*)malloc(N * sizeof(double));
    if (window == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    const double a0 = 0.54;
    const double a1 = 0.46;
    const double twoPi = 2.0 * PI;

    for (int k = 0; k < N; k++) {
        double angle = twoPi * k / (N - 1);
        window[k] = a0 - a1 * cos(angle);
    }

    return window;
}

double* multiply_arrays(const double* arr1, const double* arr2, int size) {
    double* result = (double*)malloc(size * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] * arr2[i];
    }

    return result;
}

double* subtract_arrays(const double* arr1, const double* arr2, int size) {
    double* result = (double*)malloc(size * sizeof(double));
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] - arr2[i];
    }

    return result;
}

double* FirFilterResponse(const double *input, int inputLen, const double *filter, int filterLen) {
    int outputLen = inputLen + filterLen - 1;
    double *output = (double*)malloc(outputLen * sizeof(double));
    if (output == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return NULL;
    }

    // Initialize output array to zero
    for (int i = 0; i < outputLen; i++) {
        output[i] = 0.0;
    }

    // Perform full convolution
    for (int i = 0; i < outputLen; i++) {
        for (int k = 0; k < filterLen; k++) {
            if (i - k >= 0 && i - k < inputLen) {
                output[i] += filter[k] * input[i - k];
            }
        }
    }

    return output;
}