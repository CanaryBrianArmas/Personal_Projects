# main.py

import sys
from dotenv import load_dotenv

# Load environment variables from .env file.
load_dotenv()

def print_usage():
    print("Usage: python main.py [train|eval|test]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1].lower()
    if command == "train":
        from src.train import main as train_main
        train_main()
    elif command == "eval":
        from src.eval import main as eval_main
        eval_main()
    elif command == "test":
        from src.test import main as test_main
        test_main()
    else:
        print("Invalid command. Options are: train, eval, test")
        print_usage()
