import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install('pandas')

install('scikit-learn')

install('torch')

install('pyg')
