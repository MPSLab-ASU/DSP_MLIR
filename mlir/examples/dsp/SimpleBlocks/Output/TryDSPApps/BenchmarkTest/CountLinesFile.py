import os
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
folderC = os.path.join(current_dir, 'CCode')
folderDSL = os.path.join(current_dir, 'DSP-DSL')  # Renamed this folder
folderMatlab = os.path.join(current_dir, 'Matlab')

os.makedirs('Output', exist_ok=True)

output_fileC = os.path.join(current_dir, 'Output', 'NoOfLinesInC.txt')
output_fileDSL = os.path.join(current_dir, 'Output', 'NoOfLinesInPython.txt')
output_fileMatlab = os.path.join(current_dir, 'Output', 'NoOfLinesInMatlab.txt')

def count_non_empty_linesInC(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        non_empty_code_lines = 0
        in_multiline_comment = False
        for line in lines:
            stripped_line = line.strip()
            if in_multiline_comment:
                if '*/' in stripped_line:
                    in_multiline_comment = False
                    stripped_line = stripped_line.split('*/', 1)[1]
                else:
                    continue
            if stripped_line.startswith('//'):
                continue
            if '/*' in stripped_line:
                if '*/' in stripped_line:
                    stripped_line = stripped_line.split('/*', 1)[0] + stripped_line.split('*/', 1)[1]
                else:
                    in_multiline_comment = True
                    stripped_line = stripped_line.split('/*', 1)[0]
            if stripped_line:
                non_empty_code_lines += 1
        return non_empty_code_lines

def count_valid_code_lines_in_dsl(file_path):
    valid_code_lines = 0
    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith('#'):
                valid_code_lines += 1
    return valid_code_lines

def count_valid_code_lines_in_matlab(file_path):
    valid_code_lines = 0
    with open(file_path, 'r') as file:
        in_multiline_comment = False
        for line in file:
            stripped_line = line.strip()
            if in_multiline_comment:
                if stripped_line.endswith('%}'):
                    in_multiline_comment = False
                continue
            if stripped_line.startswith('%{'):
                in_multiline_comment = True
                continue
            if stripped_line and not stripped_line.startswith('%'):
                valid_code_lines += 1
    return valid_code_lines

def count_lines_across_languages():
    line_counts = {}
    if os.path.exists(folderC):
        for filename in sorted(os.listdir(folderC)):
            file_path = os.path.join(folderC, filename)
            if os.path.isfile(file_path) and filename.endswith('.c'):
                count = count_non_empty_linesInC(file_path)
                line_counts[filename] = {'lines_in_c': count, 'lines_in_dsl': 0, 'lines_in_matlab': 0}

    # Count Python files
    if os.path.exists(folderDSL):
        for filename in sorted(os.listdir(folderDSL)):
            file_path = os.path.join(folderDSL, filename)
            if os.path.isfile(file_path) and filename.endswith('.py'):
                count = count_valid_code_lines_in_dsl(file_path)
                if filename in line_counts:
                    line_counts[filename]['lines_in_dsl'] = count
                else:
                    line_counts[filename] = {'lines_in_c': 0, 'lines_in_dsl': count, 'lines_in_matlab': 0}
    
    # Count MATLAB files
    if os.path.exists(folderMatlab):
        for filename in sorted(os.listdir(folderMatlab)):
            file_path = os.path.join(folderMatlab, filename)
            if os.path.isfile(file_path) and filename.endswith('.m'):
                count = count_valid_code_lines_in_matlab(file_path)
                if filename in line_counts:
                    line_counts[filename]['lines_in_matlab'] = count
                else:
                    line_counts[filename] = {'lines_in_c': 0, 'lines_in_dsl': 0, 'lines_in_matlab': count}
    
    return line_counts

def create_consolidated_table():
    line_counts = count_lines_across_languages()
    
    # Create a DataFrame
    df = pd.DataFrame.from_dict(line_counts, orient='index')
    
    # Reset index to make filename a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'filename'}, inplace=True)
    
    # Reorder columns
    df = df[['filename', 'lines_in_dsl', 'lines_in_c', 'lines_in_matlab']]
    
    # Fill NaN values with 0
    df.fillna(0, inplace=True)
    
    # Convert line count columns to integers
    for col in ['lines_in_dsl', 'lines_in_c', 'lines_in_matlab']:
        df[col] = df[col].astype(int)
    
    return df

def list_files_and_write_line_counts(folder, output_path, count_function, extension):
    files = sorted(os.listdir(folder))
    with open(output_path, 'w') as output:
        for filename in files:
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and filename.endswith(extension):
                line_count = count_function(file_path)
                output.write(f"{filename}: \t{line_count}\n")

if __name__ == "__main__":
     # Create the consolidated table
    consolidated_table = create_consolidated_table()

    # Save the consolidated table to a CSV file
    output_file = os.path.join('Output', 'consolidated_lines_of_code.csv')
    consolidated_table.to_csv(output_file, index=False)

    # Display the table
    print(consolidated_table)

    # Output file paths
    print(f"\nConsolidated table saved to: {output_file}")
    
    list_files_and_write_line_counts(folderC, output_fileC, count_non_empty_linesInC, '.c')
    list_files_and_write_line_counts(folderDSL, output_fileDSL, count_valid_code_lines_in_dsl, '.py')
    list_files_and_write_line_counts(folderMatlab, output_fileMatlab, count_valid_code_lines_in_matlab, '.m')