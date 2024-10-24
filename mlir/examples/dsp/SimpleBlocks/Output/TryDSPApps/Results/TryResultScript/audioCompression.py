
#fir filter design
# Input Audio → FFT → Thresholding → Quantization → Entropy Encoding → Output Compressed Audio
# Input Audio → FFT → Thresholding → Quantization → RLE Encoding → Output Compressed Audio


# give some python code using similar api -- this is for audio compression and also 
def main() {
    
    # var a10 = [ 3.2, 1.5,  0.8, 2.9,  4.5,10 , 0,5,5.5, 1.1];
    # var a10 = getRangeOfVector(3.2, 10, 1);
	var input = getRangeOfVector(0, 50000, 1);
    var nlevels = 16; #powerOf2
    var min = 0;
    var max = 8;

    var threshold = 4;

    #Get fft
    var fft10real = fft1dreal(input);
    var fft10img = fft1dimg(input);

    # print(fft10real);
    # print(fft10img);
    #Threshold
    var GetThresholdReal = threshold(fft10real , threshold);
    var GetThresholdImg = threshold(fft10img , threshold);
    # print(GetThresholdReal);
    # print(GetThresholdImg);
    #Quant
    var QuantOutReal = quantization(GetThresholdReal , nlevels, max, min);
    var QuantOutImg = quantization(GetThresholdImg , nlevels, max, min);

    print(QuantOutReal);
    print(QuantOutImg);
    #RLE
    var rLEOutReal = runLenEncoding(QuantOutReal);
    var rLEOutImg = runLenEncoding(QuantOutImg);

    # print(rLEOutReal);
    # print(rLEOutImg);

}

