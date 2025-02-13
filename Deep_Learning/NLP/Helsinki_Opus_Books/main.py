# main.py

import sys
from dotenv import load_dotenv
import warnings

# Load environment variables from .env file.
load_dotenv()

# Ignore warnings
warnings.filterwarnings("ignore")

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
    
    elif command == "predict":
        from src.predict import translate

        # Pedir input del usuario
        text = input("Introduce el texto a traducir: ")
        translation = translate(text)
        print("Traducción:", translation)

    else:
        print("Invalid command. Options are: train, eval, test")
        print_usage()
