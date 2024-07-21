import subprocess
from .utils import dmap_dir
import os

def main():
    os.chdir(dmap_dir)
    subprocess.run(["bash", "./downloadmodel.sh"])

if __name__ == '__main__':
    main()