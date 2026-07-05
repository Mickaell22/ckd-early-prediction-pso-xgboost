"""Asegura que la raiz del proyecto este en sys.path para que `import src` funcione en pytest."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
