from simple_tokenizer import SimpleTokenizer
import time

import requests
from pathlib import Path


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
    start = time.time()

    for line in lines:
        tokens = tokenizer.encode(line)
    end = time.time()
    print(f"Encoding time for {len(lines)} lines: {end - start} seconds")
    time_simple_tokenizer = end - start

    from clip_tokenizer_cpp_py import CLIPTokenizer
    import time

    tokenizer = CLIPTokenizer()
    start = time.time()

    for line in lines:
        tokens = tokenizer.encode(line)
    end = time.time()
    print(f"Encoding time for {len(lines)} lines: {end - start} seconds")
    time_clip_tokenizer_cpp = end - start

    from transformers import AutoTokenizer
    import time

    model_name = "openai/clip-vit-base-patch32"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded for model: {model_name}")

    start = time.time()
    for line in lines:
        inputs = tokenizer(line, add_special_tokens=False)
    end = time.time()
    print(f"Encoding time for {len(lines)} lines: {end - start} seconds")
    time_hf_tokenizer = end - start

    import matplotlib.pyplot as plt

    fruits = ["CLIP Tokenizer C++", "Simple Tokenizer", "Hugging Face Tokenizer"]
    sales = [time_clip_tokenizer_cpp, time_simple_tokenizer, time_hf_tokenizer]

    plt.bar(fruits, sales)
    plt.title("Encoding Time for Different Tokenizers of Moby Dick")
    plt.xlabel("Tokenizers")
    plt.ylabel("Encoding Time (seconds)")
    plt.savefig("tokenizer_encoding_time.png")
    plt.show()
