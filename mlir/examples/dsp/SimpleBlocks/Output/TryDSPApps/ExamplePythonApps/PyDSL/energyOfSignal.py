
def main() {

  #Steps:
    #calculate x[l] 
    #calculate fft : fft1 = fft(conv1)
    #then sq_abs = |abs(fft)|^2 = real^2 + img^2 
    # sum = sum(sq_abs)
    # res = gain(sum , 1/N)

    #Optimized res:
      #sq1 = input * input
      #sum1 = sum(sq1)
      

  #size 10
  # var a10 = [ 10,20,30,40,50,60,70,80,90,100];
	var input = getRangeOfVector(0, 10, 1);
  #calculate x[l] 
  #calculate fft : fft1 = fft(conv1)
  var fft_real = fft1dreal(input);
  var fft_img = fft1dimg(input);

  #then sq_abs = |abs(fft)|^2 = real^2 + img^2 
  # var sq_abs = fft_real * fft_real + fft_img * fft_img  ;
  var sq_abs = square(fft_real) + square(fft_img)  ;
  # sum = sum(sq_abs)
  var sum1 = sum(sq_abs);
  # res = gain(sum , 1/N)
  var len1 = len(input);
  var res = sum1 / len1;

  print(res);
  # var final1 = getElemAtIndx(fft_real , [6]); 
  # print(final1);



}

