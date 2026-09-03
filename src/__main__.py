import fire
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cli import RAGCLI


if __name__ == "__main__":
    try:
        fire.Fire(RAGCLI)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
