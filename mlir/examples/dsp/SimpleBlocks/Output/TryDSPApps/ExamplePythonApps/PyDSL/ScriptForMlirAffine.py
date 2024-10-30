import os
import subprocess
import time

# The script does the following
# Input : filename.py   
# Output : Mlir-affine level IR & LLVM IR for 3 cases--NoOpt, AffineOpt , AffineCanonOpt : 
# Pseudo-code:
    # Iterate for all the input-size & update the input value in file
    # Update logic -- change the 2nd parameter of line: var input = getRangeOfVector(init , Count, StepSize)
    # Run the respective commands on the file & generate 
        # AffineLevelFiles:  fileNoOpt.mlir , fileAffineOpt.mlir & fileAffineCanonOpt.mlir
        # LLVM Files:  fileNoOpt.ll , fileAffineOpt.ll & fileAffineCanonOpt.ll

#To Do:
    # CHeck for various failure cases

# Apps = "noisecancelling.py" , "lowPassFull.py" , " audioCompression.py" ,
        #  "back2backDelay.py" , "lowPassFIRFilterDesign.py" ,

# Path to the input file
input_file_path = "noisecancelling.py"  
BasePathForLLVM = "/mnt/sharedDrive/SourceCode/llvm-project/"
OutputScriptPath = "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/Results/TryResultScript/"
# OutputPath = BasePathForLLVM + "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/Results/TryResultScript/Output/"

# Construct full output path
OutputPath = os.path.join(BasePathForLLVM, OutputScriptPath, "Output")

# Check if the Output folder exists, create it if it doesn't
if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)

# Now OutputPath is ready for use
print("InputPath:{}".format(BasePathForLLVM))
print(f"OutputPath: {OutputPath}")
# exit()

# ************ Don't change unless u required 
# Define the values dictionary
inputValues = {
    "10": 10,
    # "100": 100,
    # "1K": 1000,
    # "10K": 10000,
    # "20K": 20000,
    # "30K": 30000,
    # "40K": 40000,
    # "50K": 50000,
    # "100K": 100000,
    #  "1M": 1000000,
    # "10M": 10000000,
    # "20M": 20000000,
    # "30M": 30000000,
    # "40M": 40000000,
    # "50M": 50000000,
    # "100M": 100000000,
    # "1B": 1000000000
}

commands_base = [
    # "./dsp1 lowPassFull.py -emit=mlir-affine",
    # f"./dsp1 {input_file_path} -emit=mlir-affine",
    f"{BasePathForLLVM}build/bin/dsp1 {input_file_path} -emit=mlir-affine",
    f"{BasePathForLLVM}build/bin/dsp1 {input_file_path} -emit=llvm",

]

# Define the cases
cases = [
    {"affineOpt": True, "canonOpt": True, "suffix": "fileAffineCanonOpt.mlir", "suffixll": "fileAffineCanonOpt.ll"},
    {"affineOpt": True, "canonOpt": False, "suffix": "fileAffineOpt.mlir","suffixll": "fileAffineOpt.ll"},
    {"affineOpt": False, "canonOpt": False, "suffix": "fileNoOpt.mlir" ,"suffixll": "fileNoOpt.ll"},
]


# Read the input file
with open(input_file_path, "r") as file:
    lines = file.readlines()
   
for key, value in inputValues.items():
    # Update the specific line in the file
    # print("Updating for {}".format(value))
    print("\n")
    print("{}".format(key), end="\t")
    with open(input_file_path, "w") as file:
        for line in lines:
            if line.strip().startswith("var input = getRangeOfVector("):
            # if line.strip().startswith("var N = "):
                # Replace the second parameter with the current value
                updated_line = f"    var input = getRangeOfVector(0, {value}, 0.000125);\n"
                # updated_line = f"    var N = {value + 1} ;\n"
                file.write(updated_line)
            else:
                file.write(line)
        # print(lines)

    # Iterate through the cases and run the commands
    for case in cases:
        command_mlir = commands_base[0]
        command_llvm = commands_base[1]
        if case["affineOpt"]:
            command_mlir += " -affineOpt"
            command_llvm += " -affineOpt"
        if case["canonOpt"]:
            command_mlir += " -canonOpt"
            command_llvm += " -canonOpt"
        #command_mlir += f" 2> {case['suffix']}"
        command_mlir += f" 2> {OutputPath}/{case['suffix']}" #OutputPath
        command_llvm += f" 2> {OutputPath}/{case['suffixll']}"

        commands = [
            command_mlir,
            command_llvm,
            # f"clang-17 {case['suffix']} -o fileexe -lm",
        ] 
        print(case,end="\n")
        # print("\n")
        # Iterate over each value and perform the necessary operations


        for command in commands:
            # print("running command {}".format(command))
            # os.system(command)
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
