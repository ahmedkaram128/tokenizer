sample_text = """Machine learning is a fascinating field of computer science. It allows computers to learn patterns from data instead of being explicitly programmed for every possible situation. Today, machine learning is used in many different applications, including image recognition, speech processing, recommendation systems, search engines, and natural language processing.

Natural language processing allows computers to work with human language. A language model, for example, receives a sequence of tokens and tries to predict what token is likely to come next. Although this sounds simple, modern language models can learn surprisingly complex patterns from very large amounts of text.

Before text can be given to a language model, it must first be converted into a numerical representation. This process is called tokenization. A tokenizer takes text such as "Hello, how are you?" and converts it into a sequence of tokens. These tokens may represent individual characters, parts of words, complete words, or even groups of bytes.

One popular approach is called Byte Pair Encoding, or BPE. BPE starts with small units and repeatedly combines frequently occurring pairs into larger units. For example, if the sequence "l", "o", "w" appears many times, the tokenizer may eventually learn a token representing "low". The same process can be repeated to create larger and more useful tokens.
"""

tokens = sample_text.encode("utf-8")
tokens = list(tokens)
print(tokens)
print(len(tokens))

def get_counts(ids):
    count = {}
    for pair in zip(ids, ids[1:]):
        if pair in count:
            count[pair] += 1
        else:
            count[pair] = 1
    return count


def most_frequent_pair(counts):
    return max(counts, key=counts.get)

def merge(ids, idx, pair):
    newids = []
    i = 0
    while i < len(ids):
        if len(ids) - 1 > i and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

vocab_size = 276 
num_merges = vocab_size - 256
ids = list(tokens) 

merges = {}
for i in range(num_merges):
  stats = get_counts(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx


vocab = {}

for idx in range(256):
    vocab[idx] = bytes([idx])

for pair, new_token in merge.items():
    first_byte = vocab[pair[0]]
    second_byte = vocab[pair[1]]
    new_merged_byte = first_byte + second_byte
    vocab[new_token] = new_merged_byte

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

## encoding, given string -> list of ints (tokens), check docs.txt

def encode(text): 
    tokens = list(text.encode("utf-8"))

    while len(tokens) >= 2:
        stats = get_counts(tokens)

        ## best rank is the highest possible value, (infinty)
        best_pair = None
        best_rank = float("inf")
        for pair in merges: 
            if pair in merges:
                rank = merges[pair]
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            
            if best_pair is None:
                break   
            new_token = merges[best_pair]
            merge(tokens, best_pair, new_token)
        return tokens



