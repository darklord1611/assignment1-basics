import os
import regex as re
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import BinaryIO
import multiprocessing
import time

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # intialize the byte-level vocabulary
    initial_vocab = {i: bytes([i]) for i in range(256)}
    
    # pretokenize the input file
    token_freqs = pretokenize(input_path, special_tokens, num_processes=4)
    
    # count the frequency of each pair of tokens
    pair_freqs = count_pair_freqs(token_freqs)
    
    num_merge = vocab_size - len(initial_vocab) - len(special_tokens)
    
    merges = []
    
    for _ in tqdm(range(num_merge)):
        # count the frequency of each pair of tokens
        pair_freqs = count_pair_freqs(token_freqs)
        
        best_pair = pair_freqs.most_common(1)[0][0]
        
        # merge the best pair
        token_freqs = merge_pair(best_pair, token_freqs)
        
        merges.append(best_pair)
        initial_vocab[len(initial_vocab)] = best_pair
    
    
    for special_token in special_tokens:
        initial_vocab[len(initial_vocab)] = special_token.encode("utf-8")
    
    print("Vocabulary size: ", len(initial_vocab))
    print("Merge size: ", len(merges))
    
    return initial_vocab, merges

def pretokenize(
    data_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int = 2,
):
    """
    Pre-tokenize the input file and count the frequency of each token.
    """
    with open(data_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        print(boundaries)
        
        chunks = []
        
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunks.append(chunk)
        
        with multiprocessing.Pool(num_processes) as pool:
            token_freqs_list = pool.starmap(
                pretokenize_single, [(chunk, special_tokens) for chunk in chunks]
            )
    
    # Combine the results from all processes
    total_token_freqs = sum(
        token_freqs_list, Counter()
    )
    
    return total_token_freqs

def pretokenize_single(corpus, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    delimiter = re.escape("|".join(special_tokens))
    chunks = re.split(delimiter, corpus)
    
    print(len(chunks))

    token_freqs = Counter()
    
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) == 0:
            continue
        tokens = re.finditer(PAT, chunk)
        for token in tokens:
            token_freqs[tuple(token.group().encode("utf8"))] += 1
    
    return token_freqs

def count_pair_freqs(token_freqs):
    freq_pairs = Counter()
    for token, freq in token_freqs.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            freq_pairs[pair] += freq
    return freq_pairs

def merge_pair(pair, token_freqs):
    new_freqs = Counter()
    
    for token, freq in token_freqs.items():
        new_token = []
        i = 0
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == pair:
                new_token.append(pair)
                i += 2
            else:
                new_token.append(token[i])
                i += 1
        new_freqs[tuple(new_token)] += freq
        
    return new_freqs
    

if __name__ == "__main__":
    
    # Example usage
    train_bpe(
        input_path="../data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )