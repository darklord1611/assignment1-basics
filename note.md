- word-level tokenization -> vocab size is large(~150k words) may not handle out-of-vocab words well
- byte-level tokenization -> restrict the vocab size to 256, one word can be represented by many bytes -> long sequence -> inefficient training because of long-term dependencies

We have sub-word tokenization -> iterative build the vocabulary by assign efficient number of tokens to frequent words in the sequence -> use byte-level encoding -> get the best of both worlds(out-of-vocab handling + manageable sequence length)



