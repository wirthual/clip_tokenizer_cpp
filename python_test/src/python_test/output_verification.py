import hashlib
from pathlib import Path

import requests
from clip_tokenizer_cpp_py import CLIPTokenizer
from simple_tokenizer import SimpleTokenizer
from transformers import AutoTokenizer

if __name__ == "__main__":
    file_name = "moby_dick.txt"
    if not Path(file_name).exists():
        url = "https://raw.githubusercontent.com/Mlawrence95/moby-dick/refs/heads/master/moby_dick/moby_dick.txt"

        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Open the file in binary write mode ('wb') and write the content
            with open(file_name, "wb") as f:
                f.write(response.content)
            print(f"File '{file_name}' downloaded successfully!")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")

    lines = []
    with open(file_name) as file:
        lines = [line.rstrip() for line in file]

    tokenizer = SimpleTokenizer()
    with open("tokenizer_simple.txt", "w") as f:
        for line in lines:
            tokens = tokenizer.encode(line)
            f.write(line + "\n")
            f.write(" ".join(map(str, tokens)) + "\n")

    tokenizer = CLIPTokenizer()
    with open("tokenizer_bindings.txt", "w") as f:
        for line in lines:
            tokens = tokenizer.encode(line)
            f.write(line + "\n")
            f.write(" ".join(map(str, tokens)) + "\n")

    model_name = "openai/clip-vit-base-patch32"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded for model: {model_name}")
    with open("tokenizer_hf.txt", "w") as f:
        for line in lines:
            inputs = tokenizer(line, add_special_tokens=False)
            f.write(line + "\n")
            f.write(" ".join(map(str, inputs["input_ids"])) + "\n")

    assert (
        hashlib.md5(open("tokenizer_simple.txt", "rb").read()).hexdigest()
        == hashlib.md5(open("tokenizer_bindings.txt", "rb").read()).hexdigest()
    ), "Simple tokenizer and CLIPTokenizer outputs do not match!"
    assert (
        hashlib.md5(open("tokenizer_simple.txt", "rb").read()).hexdigest()
        == hashlib.md5(open("tokenizer_hf.txt", "rb").read()).hexdigest()
    ), "Simple tokenizer and Hugging Face tokenizer outputs do not match!"
    print(
        "All tokenization outputs match across SimpleTokenizer, CLIPTokenizer, and Hugging Face tokenizer!"
    )
