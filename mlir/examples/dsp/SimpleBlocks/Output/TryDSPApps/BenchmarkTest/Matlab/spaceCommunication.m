

function main()
    % Define constants
    INPUT_LENGTH = 100000000;
    
    % Generate input vector
    input = getRangeOfVector(0, INPUT_LENGTH, 1);
    
    % Threshold
    binary_sig = thresholdUp(input, INPUT_LENGTH, 50);
    
    % Modulate
    modulated_signal = space_modulate(binary_sig, INPUT_LENGTH);
    
    % Transmit and receive (add noise)
    received_signal = transmit_and_receive(modulated_signal, INPUT_LENGTH, 1.0);
    
    % Demodulate
    demodulated_data = demodulate(received_signal, INPUT_LENGTH);
    
    % Error correction
    corrected_data = error_correction(demodulated_data);
    
    % Decode data
    decoded_data = decode_data(corrected_data);
    
    % Display first corrected byte (equivalent to printing corrected_data[8] in C)
    fprintf('%c\n', corrected_data(9));
end

% Function to generate a vector with a given range and increment
function vector = getRangeOfVector(start, length, increment)
    vector = start:increment:(start + (length - 1) * increment);
end

% Thresholding function (creates a binary string from a vector)
function output = thresholdUp(input, length, threshold)
    output = char(zeros(1, length));  % Preallocate output
    output(input > threshold) = '1';
    output(input <= threshold) = '0';
end

% Space modulation: convert binary string to modulated signal
function output = space_modulate(input, length)
    output = zeros(1, length);
    output(input == '1') = 1;
    output(input == '0') = -1;
end

% Transmit and receive (add noise based on sine of the signal)
function received_signal = transmit_and_receive(signal, length, noise_level)
    received_signal = signal + sin(signal);  % Add noise (sine-based in this case)
end

% Demodulate: convert received signal back into binary data
function demodulated_data = demodulate(signal, length)
    demodulated_data = char(zeros(1, length));
    demodulated_data(signal > 0) = '1';
    demodulated_data(signal <= 0) = '0';
end

% Error correction function
function corrected = error_correction(data)
    length = numel(data);
    corrected = char(zeros(1, length));  % Preallocate corrected array
    corrected_index = 1;
    
    for i = 1:8:length
        segment = data(i:i+7);
        count = sum(segment == '1');
        
        if mod(count, 2) == 0
            corrected(corrected_index:corrected_index+7) = segment;
        else
            corrected(corrected_index) = '0';
            corrected(corrected_index+1:corrected_index+7) = segment(2:8);
        end
        
        corrected_index = corrected_index + 8;
    end
end

% Decode binary data to ASCII characters
function decoded = decode_data(binary)
    length = numel(binary);
    decoded = char(zeros(1, length / 8));  % Preallocate decoded data array
    decoded_index = 1;
    
    for i = 1:8:length
        byte = binary(i:i+7);
        decoded(decoded_index) = char(bin2dec(byte));
        decoded_index = decoded_index + 1;
    end
end
