clc;
clear;

% Define constants
INPUT_LENGTH = 100000000;

% Generate random input data
data = randi([0 1], 1, INPUT_LENGTH);

% QAM Modulation
function symbols = qam_modulate(data)
    symbols = zeros(1, length(data)/2);
    for i = 1:2:length(data)
        bit1 = data(i);
        bit2 = data(i+1);
        
        if bit1 == 0 && bit2 == 0
            symbols((i+1)/2) = -1 - 1i;
        elseif bit1 == 0 && bit2 == 1
            symbols((i+1)/2) = -1 + 1i;
        elseif bit1 == 1 && bit2 == 0
            symbols((i+1)/2) = 1 - 1i;
        elseif bit1 == 1 && bit2 == 1
            symbols((i+1)/2) = 1 + 1i;
        end
    end
end

% QAM Demodulation
function bits = qam_demodulate(symbols)
    bits = zeros(1, length(symbols)*2);
    for i = 1:length(symbols)
        symbol = symbols(i);
        
        if symbol == -1 - 1i
            bits(2*i-1) = 0;
            bits(2*i) = 0;
        elseif symbol == -1 + 1i
            bits(2*i-1) = 0;
            bits(2*i) = 1;
        elseif symbol == 1 - 1i
            bits(2*i-1) = 1;
            bits(2*i) = 0;
        elseif symbol == 1 + 1i
            bits(2*i-1) = 1;
            bits(2*i) = 1;
        end
    end
end

% Main script
rng('shuffle'); % Seed random number generator

% Perform QAM modulation
symbols = qam_modulate(data);

% Perform QAM demodulation
bits = qam_demodulate(symbols);

% Print the 6th bit (equivalent to bits[5] in C, as MATLAB uses 1-based indexing)
disp(bits(6));