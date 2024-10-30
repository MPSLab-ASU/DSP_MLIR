def generate_tensor_string(q):
    tensor_string = "%arg0 = arith.constant dense<[[["
    for i in range(1, q*q+1):
        tensor_string += "{:.1e}".format(i*10)
        if i % q == 0 and i != q*q:
            tensor_string += "], ["
        elif i % (q*q) != 0:
            tensor_string += ", "
        else:
            tensor_string += "]"
    tensor_string += "]]> : tensor<1x{}x{}xf32>".format(q, q)
    return tensor_string

def main():
    q = 2
    tensor_string = generate_tensor_string(q)
    print(tensor_string)

if __name__ == "__main__":
    main()
