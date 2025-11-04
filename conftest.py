import sys
from pathlib import Path

# Add project root to path so pytest can find src modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))