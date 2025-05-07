INPUT_LENGTH = 400;
input = getRange(0, INPUT_LENGTH, 1);

fft = dft(input);
sq_abs = abs(fft).^2;
sum_result = sum(sq_abs);
res = sum_result / INPUT_LENGTH;

fprintf('%.6f\n', res);

function vector = getRange(start, len, step)
    vector = start + (0:len-1) * step;
end

function output = dft(input)
    N = length(input);
    output = zeros(1, N);
    for k = 1:N
        for n = 1:N
            angle = 2 * pi * (k-1) * (n-1) / N;
            output(k) = output(k) + input(n) * exp(-1i * angle);
        end
    end
end
