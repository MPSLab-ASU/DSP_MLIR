def main() {

	var input = getRangeOfVector(0, 50000, 0.000125);
        var fft_real = fft1dreal(input);
        var fft_img = fft1dimg(input);
        var sq_abs = square(fft_real) + square(fft_img)  ;
        var sum1 = sum(sq_abs);
        var len1 = len(input);
        var res = sum1 / len1;
        print(res);
}

