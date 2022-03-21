import os

id = '2794f615569a'

cwd = os.getcwd()
rootdir = os.path.join(cwd, 'l2_dropout_random_search_results', f'id-{id}')

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if (file == 'results.txt'):
            print(os.path.join(subdir, file))