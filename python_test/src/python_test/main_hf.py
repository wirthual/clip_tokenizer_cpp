from transformers import AutoTokenizer


model_name = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded for model: {model_name}")

texts = ["Hello, world! This is a test of the SimpleTokenizer."]
print(f"\nOriginal texts: {texts}")

inputs = tokenizer(texts,add_special_tokens=False)

print("\nTokenization results:")
print(f"Input IDs (token IDs): {inputs['input_ids']}")

decoded_output = [tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs['input_ids']]
print(f"\nDecoded output (with special tokens): {decoded_output}")