import os

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(BASE_DIR, 'data')
CONFIG_DIR = Path(BASE_DIR, 'config')

ENV_VARIABLES = {
    **os.environ,
}