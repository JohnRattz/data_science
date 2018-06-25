import os
import sys

PROJ_BASE_DIR = os.path.abspath(os.path.join(__file__, '../../..'))

def add_proj_py_dirs_to_path():
    """
    Adds Python source directories to the interpreter's copy of the system PATH environment variable.
    Should not be necessary when using an IDE like PyCharm.
    """
    path_dirs = ['cryptocurrencies', 'globals', 'utilities']
    path_dirs = [os.path.join(os.path.join(PROJ_BASE_DIR, path_dir), 'Python') for path_dir in path_dirs]
    for path_dir in path_dirs:
        sys.path.append(path_dir)
        ignore_dirs = ['__pycache__', '.ipynb_checkpoints']
        ignore_dirs += ['figures', 'models'] if 'cryptocurrencies' in path_dir else []
        for item in os.listdir(path_dir):
            path_subdir = os.path.join(path_dir, item)
            if os.path.isdir(path_subdir) and os.path.basename(path_subdir) not in ignore_dirs:
                sys.path.append(path_subdir)
