import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model(model_path, model_name, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move the model to the GPU or CPU
    model.eval()
    return tokenizer, model

def translate(tokenizer, model, text, device, max_length=128):
    input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids
    input_ids = input_ids.to(device)  # Move input_ids to the GPU or CPU
    
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, max_length=max_length)
    
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of the pretrained model/tokenizer")
    args = parser.parse_args()

    # Check if a GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    tokenizer, model = load_model(args.model_path, args.model_name, device)

    print("=== Classical ↔ Modern Chinese Translator ===")
    print("Enter 'quit' to exit.")
    while True:
        user_input = input(">> ")
        if user_input.lower() in ["quit", "exit"]:
            break
        output = translate(tokenizer, model, user_input, device)
        print("=>", output)

if __name__ == "__main__":
    main()
