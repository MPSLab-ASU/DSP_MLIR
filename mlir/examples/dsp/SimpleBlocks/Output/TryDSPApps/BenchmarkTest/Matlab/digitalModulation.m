INPUT_LENGTH = 100;

input = getRangeOfVector(0, INPUT_LENGTH, 0.000125);

f_sig = 500;
getMultiplier = 2 * pi * f_sig;
getSinDuration = gain(input, getMultiplier);

clean_sig = sin(getSinDuration);

binary_sig = thresholdUp(clean_sig, 0.4, 0);

modulate_symbol_real = qamModulateReal(binary_sig);
modulate_symbol_imag = qamModulateImag(binary_sig);

decode_data = qamDemodulate(modulate_symbol_real, modulate_symbol_imag);

fprintf('%.6f\n', decode_data(3));

function vector = getRangeOfVector(start, length, increment)
    vector = start + (0:length-1) * increment;
end

function output = gain(input, multiplier)
    output = input * multiplier;
end

function output = thresholdUp(input, threshold, low_value)
    output = double(input >= threshold);
    output(output < 1) = low_value;
end

function real = qamModulateReal(binary_sig)
    real = zeros(1, length(binary_sig)/2);
    for i = 1:2:length(binary_sig)
        bit1 = binary_sig(i);
        bit2 = binary_sig(i+1);
        if (bit1 == 0 && bit2 == 0) || (bit1 == 0 && bit2 == 1)
            real((i+1)/2) = -1;
        else
            real((i+1)/2) = 1;
        end
    end
end

function imag = qamModulateImag(binary_sig)
    imag = zeros(1, length(binary_sig)/2);
    for i = 1:2:length(binary_sig)
        bit1 = binary_sig(i);
        bit2 = binary_sig(i+1);
        if (bit1 == 0 && bit2 == 0) || (bit1 == 1 && bit2 == 0)
            imag((i+1)/2) = -1;
        else
            imag((i+1)/2) = 1;
        end
    end
end

function decoded = qamDemodulate(real, imag)
    decoded = zeros(1, length(real)*2);
    for i = 1:length(real)
        if real(i) == -1 && imag(i) == -1
            decoded(2*i-1:2*i) = [0 0];
        elseif real(i) == -1 && imag(i) == 1
            decoded(2*i-1:2*i) = [0 1];
        elseif real(i) == 1 && imag(i) == -1
            decoded(2*i-1:2*i) = [1 0];
        elseif real(i) == 1 && imag(i) == 1
            decoded(2*i-1:2*i) = [1 1];
        end
    end
end