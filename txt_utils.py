import numpy as np
import ast
import os

def create_local_dir(new_local_dir):
    # Create a specified local directory if it doesn't already exist
    cwd = os.getcwd()
    dir = os.path.join(cwd, new_local_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)

def unload_static_hyperparameters(txt_name):
    static_hyperparameters = {}
    with open(txt_name, 'r') as data:
        for line in data:
            if line[0] != '?':
                continue      

            static_hyperparameters = ast.literal_eval(line[2:-1])
        return static_hyperparameters

def unload_results(txt_name):
    with open(txt_name, 'r') as data:
        hyperparameters_to_stats = {}
        for line in data:
            if line[0] != '$':
                continue
            values = line.split(';')

            x = float(values[0][2:-1])
            y = float(values[1][1:-1])
            loss = float(values[2][1:-1])
            confusion_matrix = np.array(ast.literal_eval(values[3][1:-1]))

            hyperparameters_to_stats[(x, y)] = {'loss': loss, 'cm': confusion_matrix}
            
        return hyperparameters_to_stats

def unload_seeds(txt_name):
    with open(txt_name, 'r') as data:
        seeds = {}
        for line in data:
            if 'seed' not in line:
                continue
            
            line = str.lower(line)
            line = line.replace('seed', '')
            line = line.replace('\n', '')
            line = line.replace(' ', '')
            line = line.split(':')

            seeds[line[0]] = int(line[1])

        return seeds

def write_hyperparameters(name, meta_data, column_labels):
    # If the document already exists, then print a warning
    try:
        open(name, 'x')
    except:
        print('Warning: Results from a run with these settings already exists!\n')
        return
    
    # Overwrite the empty file with the hyperparameters
    with open(name, 'w') as f:
        for line in meta_data:
            print(line, file=f)
        
        print('\nBelow are the results!', file=f)
        print(column_labels, file=f)