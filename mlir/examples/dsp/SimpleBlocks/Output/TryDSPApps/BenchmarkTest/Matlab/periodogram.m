INPUT_LENGTH = 100;
input = getRangeOfVector(0, INPUT_LENGTH, 1.0);

reverse_input = reverseSignal(input);

conv_length = 2 * INPUT_LENGTH - 1;
conv1d = firFilterResponse(input, reverse_input, INPUT_LENGTH);

fft_real = dftReal(conv1d);
fft_img = dftImag(conv1d);

sq = squareMagnitude(fft_real, fft_img);

fprintf('%.6f\n', sq(3));

function vector = getRangeOfVector(start, length, increment)
    vector = start + (0:length-1) * increment;
end

function reversed = reverseSignal(input)
    reversed = input(end:-1:1);
end

function conv_out = firFilterResponse(input, filter, L)
    conv_length = 2 * L - 1;
    conv_out = zeros(1, conv_length);
    for n = 1:conv_length
        for k = 1:L
            if n - k + 1 > 0 && n - k + 1 <= L
                conv_out(n) = conv_out(n) + input(n - k + 1) * filter(k);
            end
        end
    end
end

function real = dftReal(signal)
    N = length(signal);
    real = zeros(1, N);
    for k = 0:N-1
        for n = 0:N-1
            angle = 2 * pi * k * n / N;
            real(k+1) = real(k+1) + signal(n+1) * cos(angle);
        end
    end
end

function imag = dftImag(signal)
    N = length(signal);
    imag = zeros(1, N);
    for k = 0:N-1
        for n = 0:N-1
            angle = 2 * pi * k * n / N;
            imag(k+1) = imag(k+1) - signal(n+1) * sin(angle);
        end
    end
end

function output = squareMagnitude(real, imag)
    output = real.^2 + imag.^2;
end
