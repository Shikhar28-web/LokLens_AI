import shutil
from pathlib import Path
p = Path('./data/cache')
if p.exists():
    shutil.rmtree(p)
