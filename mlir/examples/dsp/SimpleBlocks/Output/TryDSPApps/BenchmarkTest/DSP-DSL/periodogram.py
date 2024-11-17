
def main() {

  #Steps:
    #calculate x[l] , x[-l]
    #calculate conv1d of x[l] , x[-l] ie, conv1 = conv(x[l] , x[-l])
    #calculate fft : res = fft(conv1)
    #then periodogram = |abs(fft)|^2 = real^2 + img^2 

    #Another way:
      #pad x[l] & x[-l] with zeroes
      #calculate fft of x[l] & x[-l] ie, fft_x , fft_reverse_x
      #multiply them to get final real ans : fft_x * fft_reverse_x

  #size 10
  # var a10 = [ 10,20,30,40,50,60,70,80,90,100];
	var input = getRangeOfVector(0, 30000, 1);
  # var input = [1,2,3,4];
  # print(a10);

  #Get x[-l] ie, reverseInput & 
  var reverse_input = reverseInput(input);
  # y[n] = sum(h(k) . x(n-k)) k=0 to N-1 & 0<= n < N
  var conv1d = FIRFilterResponse(input, reverse_input);
  # var fft_real = fft1DRealSymm(conv1d); #fft1DRealSymm
  var fft_real = fft1dreal(conv1d);
  var fft_img = fft1dimg(conv1d);
  var sq = fft_real * fft_real + fft_img * fft_img;
  print(sq);
  var final1 = getElemAtIndx(sq , [2]); 
  print(final1);
}

