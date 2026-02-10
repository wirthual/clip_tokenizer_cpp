import gzip
import html
from functools import lru_cache

import ftfy
import regex as re

import importlib.resources as pkg_resources
import time

import requests
from pathlib import Path


@lru_cache()
def default_bpe():
    return pkg_resources.files("python_test").joinpath("bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )

    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text


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
    import numpy as np

    fruits = ['CLIP Tokenizer C++', 'Simple Tokenizer',  'Hugging Face Tokenizer']
    sales = [time_clip_tokenizer_cpp,time_simple_tokenizer, time_hf_tokenizer]

    plt.bar(fruits, sales)
    plt.title('Encoding Time for Different Tokenizers of Moby Dick')
    plt.xlabel('Tokenizers')
    plt.ylabel('Encoding Time (seconds)')
    plt.savefig('tokenizer_encoding_time.png')
    plt.show()
