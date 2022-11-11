from pathlib import Path
import pandas as pd

from .narrative import Narratives

# creates a folder, does not overwrite if exists
def create_folder(path):
    # create a temp folder for intermittent files including parents
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def read_jsonl_chunks(path):
    print('Loading file {} ...'.format(path))
    chunks = pd.read_json(path, lines=True, chunksize=10000)
    jfile = None
    for chunk in chunks:
        if jfile is None:
            jfile = chunk
        else:
            jfile = pd.concat([jfile, chunk], axis=0)

    jfile = jfile.reset_index(drop=True)
    return jfile