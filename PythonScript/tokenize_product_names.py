import csv
from transformers import AutoTokenizer

def tokenize_product_names_from_csv():
    input_file = "C:\\RPA\\Image\\Target\\input.csv"
    output_file = "C:\\RPA\\Image\\Target\\output.csv"
    tokenizer_name = "jhgan/ko-sroberta-multitask"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load product names from CSV
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        product_names = [row[0] for row in reader]
    
    # Tokenize product names
    tokenized_names = [tokenizer.tokenize(name) for name in product_names]
    
    # Remove "##" from the tokens
    tokenized_names = [[token.replace("##", "") for token in name] for name in tokenized_names]
    
    # Save tokenized names to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for tokens in tokenized_names:
            writer.writerow(tokens)