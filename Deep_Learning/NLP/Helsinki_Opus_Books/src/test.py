# src/test.py

from src.model import load_model_and_tokenizer

def test_model_inference():
    model, tokenizer = load_model_and_tokenizer()
    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Input:", input_text)
    print("Translation:", translation)

def main():
    print("Running tests...")
    test_model_inference()
    print("All tests completed successfully.")

if __name__ == "__main__":
    main()
