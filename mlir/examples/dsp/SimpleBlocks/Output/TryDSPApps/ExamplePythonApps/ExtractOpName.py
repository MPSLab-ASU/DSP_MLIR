import re
import os

fileNamePath = "mlir/examples/dsp/SimpleBlocks/include/toy/Ops.td"
# BasePathForLLVM = "/mnt/sharedDrive/SourceCode/llvm-project/"
# OutputScriptPath = "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/Results/TryResultScript/"
BasePathForLLVM = "/home/local/ASURITE/apkhedka/ForLLVM/"
fileName = BasePathForLLVM + fileNamePath
print(fileName)
# Create 'Output' folder if it doesn't exist
os.makedirs('Output', exist_ok=True)

OutputFile = 'Output/OpsNameDump.txt'

# Read text from file
with open(fileName, 'r') as file:
    text = file.read()

# Regular expression to find Op class names and operation names
pattern = r'def (\w+) : Dsp_Op<"(\w+)"'

# Find all matches
matches = re.findall(pattern, text)

# Print results

# Write results to file
with open(OutputFile, 'w') as file:
    for match in matches:
        # file.write(f"Op class: {match[0]}, Operation name: {match[1]}\n") 
        file.write(f"{match[1]}, ")
        # file.write(", ".join(f"{match[1]}"))
# for match in matches:
#     countOfOps = countOfOps +1
#     print(f"Op class: {match[0]}, Operation name: {match[1]}")

print("TotalOps Count= {}".format(len(matches)))
# print(matches)
#Already existing = 10 , total = 46 , 
#