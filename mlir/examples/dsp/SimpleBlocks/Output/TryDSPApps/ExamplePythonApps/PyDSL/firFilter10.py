
def main() {
  # var a = [10,20,30];
  
  #size = 10
#   var input<2,5> = [10,20,30,40,50,60, 70, 80, 90 , 100];
#   var filter<2,5> = [10,20,30,40,50,60, 70, 80, 90 , 100];
  # var input = [10,20,30,40,50,60, 70, 80, 90 , 100];
  # var filter = [10,20,30,40,50,60, 70, 80, 90 , 100];
	# var input = getRangeOfVector(1, 10, 1);
  var input = [1,2,3,4];
  var reverse1 = reverseInput(input);

  # var filter = [4,3,2,1];
  var output = FIRFilterResponse(input , reverse1);
  # var output = FIRFilterYSymmOptimized(input , reverse1);
  # var fft_real = fft1dreal(output);
  # var fft_real = fft1DRealSymm(output);
  var fft_img = fft1DimgConjSymm(output);
  # var fft_img = fft1dimg(output);
  # print(fft_real);
  print(fft_img);

  # var lenOut = len(output);
  # var reverse1 = reverseInput(input);
  # var lenIn = len(input);
  # var padlen = lenOut - lenIn;

  # var pad1 = padding(input , 0 , 3);
  # print(pad1);
  # var output = FIRFilterResSymmOptimized(input , filter);
  # print(output);
  # print(len1);
  # print(reverse1);
  # print(pad1);
}