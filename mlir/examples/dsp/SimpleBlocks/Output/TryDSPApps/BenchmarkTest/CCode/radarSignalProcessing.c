#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#define PI 3.1415926
#define INPUT_LENGTH 100000000

// Function declarations
double *getRangeOfVector(double start, double end, double step);
double complex *beam_form(int antennas, double input_fc, double *input, double *weights, int input_length);
double *abs_array(double complex *array, int length);
int argmax(double *array, int length);
double *lowPassFIRFilter(double wc, int length);
double *highPassFIRFilter(double wc, int length);
double *hamming(int length);
double *elementWiseMultiply(double *array1, double *array2, int length);
double *subtract(double *array1, double *array2, int length);
double *FIRFilterResponse(double *input, double *filter, int input_length, int filter_length);
double getElemAtIndx(double *array, int index);

int main() {
    // Parameters
    int antennas = 4;
    double input_fc = 5;
    int N = 101;
    
    // Generate input vector
    double *input = getRangeOfVector(0, INPUT_LENGTH, 1);
    
    // Generate weights vector
    int weights_length = (180 - (-90)) / 1 + 1;
    double *weights = getRangeOfVector(-90, 180, 1);

    // Beamforming
    double complex *signal = beam_form(antennas, input_fc, input, weights, INPUT_LENGTH);
    double *b1 = abs_array(signal, INPUT_LENGTH);
    
    // Power profile calculation
    double *power_profile = elementWiseMultiply(b1, b1, INPUT_LENGTH);
    int power_angle_max_idx = argmax(power_profile, INPUT_LENGTH);
    double power_angle_max_ele = power_profile[power_angle_max_idx];

    // Filter design parameters
    double fc1 = 1000;
    double fc2 = 7500;
    double Fs = 8000;

    // Low-pass filter
    double wc1 = 2 * PI * fc1 / Fs;
    double *filter1 = lowPassFIRFilter(wc1, N);
    double *hamming_window = hamming(N);
    double *filter_hamming_1 = elementWiseMultiply(filter1, hamming_window, N);

    // High-pass filter
    double wc2 = 2 * PI * fc2 / Fs;
    double *filter2 = highPassFIRFilter(wc2, N);
    double *filter_hamming_2 = elementWiseMultiply(filter2, hamming_window, N);

    // Band-pass filter
    double *bpf = subtract(filter_hamming_2, filter_hamming_1, N);
    
    // FIR filter response
    double *firFilterResponse = FIRFilterResponse(power_profile, bpf, INPUT_LENGTH, N);
    
    // Get final result
    if (INPUT_LENGTH > 2) {
        double final_value = getElemAtIndx(firFilterResponse, 2);
        printf("%f\n", final_value);
    } else {
        fprintf(stderr, "Input length is too small to retrieve the element at index 2.\n");
    }

    // Free allocated memory
    free(input);
    free(weights);
    free(signal);
    free(b1);
    free(power_profile);
    free(filter1);
    free(hamming_window);
    free(filter_hamming_1);
    free(filter2);
    free(filter_hamming_2);
    free(bpf);
    free(firFilterResponse);

    return 0;
}

// Function implementations

double *getRangeOfVector(double start, double end, double step) {
    if (step <= 0) {
        fprintf(stderr, "Step must be positive\n");
        exit(EXIT_FAILURE);
    }
    
    int size = (int)((end - start) / step) + 1;
    
    if (size <= 0) {
        fprintf(stderr, "Invalid range for vector\n");
        exit(EXIT_FAILURE);
    }

    double *vector = malloc(size * sizeof(double));
    
    if (vector == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < size; i++) {
        vector[i] = start + i * step;
    }
    
    return vector;
}

double complex *beam_form(int antennas, double input_fc, double *input, double *weights, int input_length) {
    if (antennas <= 0 || weights == NULL || input == NULL || input_length <= 0) {
        fprintf(stderr, "Invalid parameters for beam_form function\n");
        exit(EXIT_FAILURE);
    }

    double complex *signal = malloc(input_length * sizeof(double complex));
    
    if (signal == NULL) {
        fprintf(stderr, "Memory allocation failed for signal\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < input_length; i++) {
        signal[i] = 0;
        for (int j = 0; j < antennas; j++) {
            signal[i] += weights[j] * cexp(I * (2.0 * PI * input_fc * input[i] + j * PI / 2));
        }
    }
    
    return signal;
}

double *abs_array(double complex *array, int length) {
   if (length <= 0 || array == NULL) {
       fprintf(stderr,"Invalid parameters for abs_array function\n");
       exit(EXIT_FAILURE); 
   }

   double *result = malloc(length * sizeof(double));
   
   if (result == NULL) {
       fprintf(stderr,"Memory allocation failed for result in abs_array\n");
       exit(EXIT_FAILURE); 
   }

   for (int i = 0; i < length; i++) {
       result[i] = cabs(array[i]);
   }
   
   return result;
}

int argmax(double *array, int length) {
   if (length <= 0 || array == NULL) {
       fprintf(stderr,"Invalid parameters for argmax function\n");
       exit(EXIT_FAILURE); 
   }

   int max_idx = 0;
   for (int i = 1; i < length; i++) {
       if (array[i] > array[max_idx]) {
           max_idx = i;
       }
   }
   
   return max_idx;
}

double* lowPassFIRFilter(double wc,int length){
   if(length <= 0){
      fprintf(stderr,"Invalid filter length in lowPassFIRFilter\n");
      exit(EXIT_FAILURE); 
   }

   double* filter=malloc(length*sizeof(double));
   
   if(filter==NULL){
      fprintf(stderr,"Memory allocation failed for low pass filter\n");
      exit(EXIT_FAILURE); 
   }

   int mid=(length-1)/2;

   for(int n=0;n<length;n++){
      if(n==mid){
         filter[n]=wc/PI;
      }else{
         filter[n]=sin(wc*(n-mid))/(PI*(n-mid));
      }
   }
   
   return filter;
}

double* highPassFIRFilter(double wc,int length){
   if(length <= 0){
      fprintf(stderr,"Invalid filter length in highPassFIRFilter\n");
      exit(EXIT_FAILURE); 
   }

   double* filter=malloc(length*sizeof(double));
   
   if(filter==NULL){
      fprintf(stderr,"Memory allocation failed for high pass filter\n");
      exit(EXIT_FAILURE); 
   }

   int mid=(length-1)/2;

   for(int n=0;n<length;n++){
      if(n==mid){
         filter[n]=1-(wc/PI);
      }else{
         filter[n]=-sin(wc*(n-mid))/(PI*(n-mid));
      }
   }
   
   return filter;
}

double* hamming(int length){
   if(length <= 0){
      fprintf(stderr,"Invalid window length in hamming\n");
      exit(EXIT_FAILURE); 
   }

   double* window=malloc(length*sizeof(double));
   
   if(window==NULL){
      fprintf(stderr,"Memory allocation failed for hamming window\n");
      exit(EXIT_FAILURE); 
   }

   for(int i=0;i<length;i++){
      window[i]=0.54-0.46*cos(2*PI*i/(length-1));
   }
   
   return window;
}

double* elementWiseMultiply(double* array1,double* array2,int length){
   if(length <= 0 || array1 == NULL || array2 == NULL){
      fprintf(stderr,"Invalid parameters in elementWiseMultiply\n");
      exit(EXIT_FAILURE); 
   }

   double* result=malloc(length*sizeof(double));
   
   if(result==NULL){
      fprintf(stderr,"Memory allocation failed for element wise multiplication result\n");
      exit(EXIT_FAILURE); 
   }

   for(int i=0;i<length;i++){
      result[i]=array1[i]*array2[i];
   }
   
   return result;
}

double* subtract(double* array1,double* array2,int length){
   if(length <= 0 || array1 == NULL || array2 == NULL){
      fprintf(stderr,"Invalid parameters in subtract function\n");
      exit(EXIT_FAILURE); 
   }

   double* result=malloc(length*sizeof(double));
   
   if(result==NULL){
      fprintf(stderr,"Memory allocation failed for subtraction result\n");
      exit(EXIT_FAILURE); 
   }

   for(int i=0;i<length;i++){
      result[i]=array1[i]-array2[i];
   }
   
   return result;
}

double* FIRFilterResponse(double* input,double* filter,int input_length,int filter_length){
     if(input_length <= 0 || filter_length <= 0 || input == NULL || filter == NULL){
         fprintf(stderr,"Invalid parameters in FIRFilterResponse function\n");
         exit(EXIT_FAILURE); 
     }

     double* response=malloc(input_length*sizeof(double));
     
     if(response==NULL){
         fprintf(stderr,"Memory allocation failed for FIR Filter Response\n");
         exit(EXIT_FAILURE); 
     }

     for(int i=0;i<input_length;i++){
         response[i]=0;
         for(int j=0;j<filter_length && i-j>=0;j++){
             response[i]+=input[i-j]*filter[j];
         }
     }
     
     return response;
}

double getElemAtIndx(double* array,int index){
     return array[index];
}