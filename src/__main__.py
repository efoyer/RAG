import fire
import sys
from .cli import RAGCLI


if __name__ == "__main__":
    try:
        fire.Fire(RAGCLI)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
