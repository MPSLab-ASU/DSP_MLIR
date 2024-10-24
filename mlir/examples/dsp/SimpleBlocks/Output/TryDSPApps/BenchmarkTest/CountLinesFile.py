import os

# folder1 = "./CCode"  # Replace with your folder path
# Get the current Python file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the folder path relative to the current directory
folderC = os.path.join(current_dir, 'CCode')
folderPy = os.path.join(current_dir, 'PyDSL')

os.makedirs('Output', exist_ok=True)

# Specify the output file path
output_fileC = os.path.join(current_dir, 'Output', 'NoOfLinesInC.txt')
output_filePy = os.path.join(current_dir, 'Output', 'NoOfLinesInPy.txt')

def count_non_empty_linesInC(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        non_empty_lines = [line for line in lines if line.strip()]
        return len(non_empty_lines)

def count_valid_code_linesInPyFile(file_path):
    valid_code_lines = 0

    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            # Check if the line is not empty and does not start with a comment
            if stripped_line and not stripped_line.startswith('#'):
                valid_code_lines += 1

    return valid_code_lines
    

def list_files_and_write_line_counts(folder, output_path):
    # List files in the folder and sort them by filename
    files = sorted(os.listdir(folder))
    with open(output_path, 'w') as output:
        for filename in files:
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and filename.endswith('.c'):  # Check if it's a text file
                line_count = count_non_empty_linesInC(file_path)
                output.write(f"{filename}: \t{line_count} \n")

def list_files_and_write_line_countsPy(folder, output_path):
    # List files in the folder and sort them by filename
    files = sorted(os.listdir(folder))
    with open(output_path, 'w') as output:
        for filename in files:
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and filename.endswith('.py'):  # Check if it's a text file
                line_count = count_valid_code_linesInPyFile(file_path)
                output.write(f"{filename}: \t{line_count}\n")


# Call the function
list_files_and_write_line_counts(folderC, output_fileC)
list_files_and_write_line_countsPy(folderPy, output_filePy)

