
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
    #Threshold --y_threshold[n] = a[i]  if a[i] >= threshld or, a[i] <= -threshld
                  # = 0 , else
    var GetThresholdReal = threshold(fft10real , threshold);
    var GetThresholdImg = threshold(fft10img , threshold);
    # # print(GetThresholdReal);
    # # print(GetThresholdImg);
    # #Quant: y_quantized[i] = Round(a[i] - min) / step) * step + min
        # where, step = (max-min)/NoOfLevels
    var QuantOutReal = quantization(GetThresholdReal , nlevels, max, min);
    var QuantOutImg = quantization(GetThresholdImg , nlevels, max, min);

    # print(QuantOutReal);
    # print(QuantOutImg);
    # #RLE: y_rle[i] =  x[i] , if x[i] != x[i-1] , 1<=i<n
                # CountOfXi , at n<=i < 2n -1
    var rLEOutReal = runLenEncoding(QuantOutReal);
    var rLEOutImg = runLenEncoding(QuantOutImg);

    # get elem at given indx of the vector
    var final1 = getElemAtIndx(rLEOutReal , [6]); 
  var final2 = getElemAtIndx(rLEOutImg , [7]);
  print(final1);
  print(final2);
    # print(rLEOutReal);
    # print(rLEOutImg);

}

