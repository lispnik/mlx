# Examples

Each example is a standalone file. Run it from the project root:

```sh
sbcl --load examples/mnist.lisp --eval '(mnist:main)' --quit
sbcl --load examples/transformer.lisp --eval '(char-gpt:main)' --quit
```

Datasets are downloaded on first use to `~/.cache/mlx-cl/datasets/`.

## `mnist.lisp`: digit classification

Trains an MLP and a small CNN (conv, max-pool, dropout) on MNIST, using
`nn:value-and-grad`, AdamW and `with-scope`. It reports test accuracy after
each epoch.

On an M3, two epochs give about 97.5% for the MLP (0.4 s per epoch) and 98.9%
for the CNN (4 s per epoch).

## `transformer.lisp`: a character-level GPT

A 3.2M-parameter decoder-only transformer trained from scratch on Tiny
Shakespeare. It uses causal `multi-head-attention`, pre-norm blocks, learned
positional embeddings, AdamW with a warmup and cosine schedule, and gradient
clipping, then samples text.

On an M3, 1500 steps take about 3 minutes and bring validation loss from 4.35
to 1.55 nats per character:

```
ROMEO:
POMPEY:
O, he something of his song, but my love,
More should be doth cause the gentle reign.
```

## Pretrained language models

For running pretrained Hugging Face models (Llama, Qwen2, Mistral, Phi-3 and
Gemma), see `mlx.llm` in the main README, or the CLI:

```sh
bin/mlx-cl generate -m mlx-community/gemma-3-1b-it-4bit "Explain monads briefly"
```
