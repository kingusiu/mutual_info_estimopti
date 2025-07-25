import sys
from pathlib import Path

# Add src to sys.path for development/demo purposes
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mi_estimopt import example_function

def main():
    print("Demo: example_function(42) =", example_function(42))

if __name__ == "__main__":
    main()
