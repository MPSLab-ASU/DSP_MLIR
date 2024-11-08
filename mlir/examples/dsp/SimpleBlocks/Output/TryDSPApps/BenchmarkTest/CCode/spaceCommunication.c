#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_LENGTH 100000000

// Function prototypes
double *getRangeOfVector(double start, int length, double increment);
void thresholdUp(const double *input, int length, double threshold,
                 char *output);
void space_modulate(const char *input, int *output, int length);
void transmit_and_receive(const int *signal, double *received_signal,
                          int length, double noise_level);
void demodulate(const double *signal, char *demodulated_data, int length);
void error_correction(const char *data, char *corrected);
void decode_data(const char *binary, char *decoded);

// Function implementations
void thresholdUp(const double *input, int length, double threshold,
                 char *output) {
  for (int i = 0; i < length; i++) {
    output[i] = (input[i] > threshold) ? '1' : '0';
  }
  output[length] = '\0';
}

void space_modulate(const char *input, int *output, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = (input[i] == '1') ? 1 : -1;
  }
}

void transmit_and_receive(const int *signal, double *received_signal,
                          int length, double noise_level) {
  for (int i = 0; i < length; i++) {
    double noise = sin(signal[i]);
    received_signal[i] = signal[i] + noise;
  }
}

void demodulate(const double *signal, char *demodulated_data, int length) {
  for (int i = 0; i < length; i++) {
    demodulated_data[i] = (signal[i] > 0) ? '1' : '0';
  }
  demodulated_data[length] = '\0';
}

void error_correction(const char *data, char *corrected) {
  int length = strlen(data);
  int corrected_index = 0;
  for (int i = 0; i < length; i += 8) {
    int count = 0;
    for (int j = 0; j < 8; j++) {
      if (data[i + j] == '1')
        count++;
    }
    if (count % 2 == 0) {
      strncpy(&corrected[corrected_index], &data[i], 8);
    } else {
      corrected[corrected_index] = '0';
      strncpy(&corrected[corrected_index + 1], &data[i + 1], 7);
    }
    corrected_index += 8;
  }
  corrected[corrected_index] = '\0';
}

void decode_data(const char *binary, char *decoded) {
  int length = strlen(binary);
  int decoded_index = 0;
  for (int i = 0; i < length; i += 8) {
    char byte[9];
    strncpy(byte, &binary[i], 8);
    byte[8] = '\0';
    decoded[decoded_index++] = (char)strtol(byte, NULL, 2);
  }
  decoded[decoded_index] = '\0';
}

double *getRangeOfVector(double start, int length, double increment) {
  double *vector = malloc(length * sizeof(double));
  if (!vector) {
    perror("Memory allocation failed in getRangeOfVector");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < length; i++) {
    vector[i] = start + i * increment;
  }
  return vector;
}

int main() {
  // Generate input vector
  double *input = getRangeOfVector(0, INPUT_LENGTH, 1);
  if (!input) {
    perror("Memory allocation failed for input");
    return EXIT_FAILURE;
  }

  // Threshold
  char *binary_sig = malloc(INPUT_LENGTH + 1);
  if (!binary_sig) {
    perror("Memory allocation failed for binary_sig");
    free(input);
    return EXIT_FAILURE;
  }
  thresholdUp(input, INPUT_LENGTH, 50, binary_sig);

  // Modulate
  int *modulated_signal = malloc(INPUT_LENGTH * sizeof(int));
  if (!modulated_signal) {
    perror("Memory allocation failed for modulated_signal");
    free(input);
    free(binary_sig);
    return EXIT_FAILURE;
  }
  space_modulate(binary_sig, modulated_signal, INPUT_LENGTH);

  // Transmit and receive (add noise)
  double *received_signal = malloc(INPUT_LENGTH * sizeof(double));
  if (!received_signal) {
    perror("Memory allocation failed for received_signal");
    free(input);
    free(binary_sig);
    free(modulated_signal);
    return EXIT_FAILURE;
  }
  transmit_and_receive(modulated_signal, received_signal, INPUT_LENGTH, 1.0);

  // Demodulate
  char *demodulated_data = malloc(INPUT_LENGTH + 1);
  if (!demodulated_data) {
    perror("Memory allocation failed for demodulated_data");
    free(input);
    free(binary_sig);
    free(modulated_signal);
    free(received_signal);
    return EXIT_FAILURE;
  }
  demodulate(received_signal, demodulated_data, INPUT_LENGTH);

  // Error correction
  char *corrected_data = malloc(INPUT_LENGTH + 1);
  if (!corrected_data) {
    perror("Memory allocation failed for corrected_data");
    free(input);
    free(binary_sig);
    free(modulated_signal);
    free(received_signal);
    free(demodulated_data);
    return EXIT_FAILURE;
  }
  error_correction(demodulated_data, corrected_data);

  // Decode data
  char *decoded_data = malloc((INPUT_LENGTH / 8) + 1);
  if (!decoded_data) {
    perror("Memory allocation failed for decoded_data");
    free(input);
    free(binary_sig);
    free(modulated_signal);
    free(received_signal);
    free(demodulated_data);
    free(corrected_data);
    return EXIT_FAILURE;
  }
  decode_data(corrected_data, decoded_data);

  printf("%c", corrected_data[8]);

  // Free allocated memory
  free(input);
  free(binary_sig);
  free(modulated_signal);
  free(received_signal);
  free(demodulated_data);
  free(corrected_data);
  free(decoded_data);

  return 0;
}