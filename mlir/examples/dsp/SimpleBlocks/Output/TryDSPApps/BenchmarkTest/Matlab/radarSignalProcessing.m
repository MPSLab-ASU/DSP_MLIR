% Constants
PI = pi;
INPUT_LENGTH = 10000;

% Function prototypes
input = linspace(0, (INPUT_LENGTH - 1) * 0.000125, INPUT_LENGTH); % Using linspace instead of getrangeofvector
weights = linspace(-90, 180, 4); % Example antenna weights
antennas = 4;
input_fc = 5;
N = 101;
fc1 = 1000;
fc2 = 7500;
Fs = 8000;

% Generate beamformed signal
signal = beamForm(antennas, input_fc, input, weights);

% Compute absolute values and power profile
b1 = abs(signal);
power = b1 .^ 2; % element-wise square instead of power_profile

% Low-pass and high-pass FIR filters with Hamming window
wc1 = 2 * PI * fc1 / Fs;
filter1 = lowPassFIRFilter(wc1, N);
filter_hamming_1 = filter1 .* hamming(N, 'symmetric')'; % Using 'symmetric' Hamming window

wc2 = 2 * PI * fc2 / Fs;
filter2 = highPassFIRFilter(wc2, N);
filter_hamming_2 = filter2 .* hamming(N, 'symmetric')'; % Using 'symmetric' Hamming window

% Band-pass filter by subtracting the filters
bpf = filter_hamming_2 - filter_hamming_1;

% Apply FIR filter to the power profile (use full convolution)
firFilterResponse = conv(power, bpf, 'full'); % Use 'full' to match C code

% Output final value at the 10000th index (adjust if necessary)
final = firFilterResponse(2); % Adjust to match desired index in C code
fprintf('final: %f\n', final);

% Functions

function output = beamForm(antennas, frequency, time, weights)
    phase_var = 2 * pi * frequency;
    signal = zeros(antennas, length(time));

    for i = 1:antennas
        iter_args = (i - 1) * pi / 4.0;
        signal(i, :) = sin(time * phase_var + iter_args);
    end

    output = sum(signal .* weights', 1); % Beamforming by weighted summation
end

function output = lowPassFIRFilter(wc, N)
    midIndex = (N - 1) / 2;
    output = zeros(1, N);

    for i = 1:N
        if i == midIndex + 1
            output(i) = wc / pi;
        else
            output(i) = sin(wc * (i - midIndex - 1)) / (pi * (i - midIndex - 1));
        end
    end
end

function output = highPassFIRFilter(wc, N)
    midIndex = (N - 1) / 2;
    output = zeros(1, N);

    for i = 1:N
        if i == midIndex + 1
            output(i) = 1 - wc / pi;
        else
            output(i) = -sin(wc * (i - midIndex - 1)) / (pi * (i - midIndex - 1));
        end
    end
end
