
Script Steps:
	i) Input : List of files 
		--a) each file has different inputs[5 diff IP Size- 10, 100 , 1K, 10K, 100K ] & needs to be commented and 
		--b) Each file should generate 2 versions of affine-level 
			-- With Affine Opt , With Affine & Canonical Opt

		--c) For each of the 2 output, we need to compile it to llvm 
		--d) From each of the 2 llvm , we need to get executable
		--e) For each of the 2 exe, we need to get the time 
		--f) Total 10 

	Pseudo code:
		for each IPSize :
			input = IPSize
			NoCanonical, Canonical = Call 2 binaries on the file -mlir-affine

			NoCanonicalLLvm , CanonicalLLvm = Call convert to llvm:
				emit=llvm

			NoCanonicalExe , CanonicalExe = clang-17 NoCanonicalLLvm.ll -o 

			TimeNoCanonical , TimeCanonical = time ./NoCanonicalExe 






