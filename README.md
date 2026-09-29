# mlx — Common Lisp bindings to Apple MLX

[![CI](https://github.com/lispnik/mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/lispnik/mlx/actions/workflows/ci.yml)

Complete SBCL bindings to [mlx-c](https://github.com/ml-explore/mlx-c), the C API
of Apple's MLX array framework: lazy n-dimensional arrays on the Apple Silicon GPU
and CPU, automatic differentiation, vectorization, graph compilation, custom
Metal kernels, and safetensors/GGUF I/O.

On top of that it provides:

- **`mlx.nn` and `mlx.optimizers`.** Ports of Python MLX's neural-network
  and optimizer libraries that match them numerically.
- **`mlx/llm`.** Runs Hugging Face Llama, Qwen2/3, Mistral, Phi-3 and
  Gemma 2/3 models, and the mixture-of-experts Mixtral, Qwen2-MoE,
  Qwen3-MoE and OLMoE. It produces the same tokens as mlx-lm at the same
  speed.

Runnable examples, MNIST and a character-level GPT trained from scratch,
are in [`examples/`](examples/).

```lisp
(defpackage :demo (:use :cl) (:local-nicknames (:mx :mlx) (:random :mlx.random)))
(in-package :demo)

(let* ((a (mx:from-lisp '((1.0 2.0) (3.0 4.0))))
       (b (mx:add (mx:matmul a a) 1)))          ; lazy: nothing computed yet
  (mx:to-lisp b))                                ; => #2A((8.0 11.0) (16.0 23.0))

(funcall (mx:grad (lambda (x) (mx:sum (mx:square x)))) (mx:from-lisp '(1.0 2.0)))
;; => #<MLX-ARRAY float32 (2) array([2, 4], dtype=float32)>
```

## Requirements

- Apple Silicon Mac (the bindings rely on the arm64 calling convention; see below)
- `brew install mlx-c` (tested with mlx-c 0.6 / MLX 0.32)
- SBCL and [ocicl](https://github.com/ocicl/ocicl)

```sh
ocicl install          # cffi, trivial-garbage, fiveam, clingon (from ocicl.csv)
make test              # FiveAM suites (MLX_CL_TEST_DEVICE=cpu forces the CPU;
                       # MLX_CL_TEST_MODELS=1 adds tests against real model weights;
                       # MLX_CL_TEST_ALL_MODELS=1 adds seven more families (~9 GB);
                       # MLX_CL_TEST_EXACT=1 also demands full-length greedy
                       # agreement with mlx-lm, which only holds on the GPU
                       # generation and MLX version the fixtures came from:
                       # an M3, MLX 0.32.1)
make cli               # builds bin/mlx-cl
```

If libmlxc isn't in a standard location, set `MLX_C_LIBRARY=/path/to/libmlxc.dylib`.

## Layout

| Path | Contents |
| --- | --- |
| `tools/generate.lisp` | Parses the mlx-c headers and generates the two files below |
| `src/ffi/bindings.lisp` | **Generated.** All 619 C functions as `cffi:defcfun`, plus enums and handle types (package `mlx-ffi`) |
| `src/generated/ops.lisp` | **Generated.** 305 high-level ops as `define-op` specs, plus axis-family dispatchers |
| `src/op.lisp` | `define-op`: argument conversion, result slots, status checking |
| `src/*.lisp` | Hand-written layer: arrays, devices and streams, closures and transforms, I/O, kernels |
| `src/nn/` | `mlx.nn` and `mlx.optimizers` |
| `src/llm/` | Tokenizer, the configurable decoder model, generation, Hub download (system `mlx/llm`) |
| `src/symreg/` | Symbolic regression by genetic programming on the GPU (system `mlx/symreg`) |
| `examples/` | MNIST and a character-level GPT, runnable from the project root |
| `tests/` | FiveAM suites (`mlx/tests`, `mlx/llm-tests`, `mlx/symreg-tests`) and reference fixtures |
| `cli/main.lisp` | clingon command-line driver |

To regenerate after upgrading mlx-c, run `make generate`, or
`sbcl --script tools/generate.lisp /path/to/include/mlx/c`.

Two guards protect against mlx-c changes:

- `make check-generated` fails if the committed generated files differ from
  what the installed headers produce. CI runs it on every push and weekly.
- The generator refuses to run (exit 2) if any struct the bindings pass by
  value changes layout, or if the dtype enum changes. The failure shows the
  expected and found layouts. The `:mlx.abi` test suite pins each
  convention against the live library.

## Packages

| Package | Mirrors | Examples |
| --- | --- | --- |
| `mlx` | `mlx.core` | `add` `matmul` `sum` `reshape` `where` `conv2d` `quantize` `grad` `compile` |
| `mlx.linalg` | `mlx.core.linalg` | `inv` `solve` `qr` `svd` `eigh` `cholesky` `norm` |
| `mlx.fft` | `mlx.core.fft` | `fft` `rfft` `fft2` `fftn` `irfftn` `fftshift` |
| `mlx.random` | `mlx.core.random` | `seed` `key` `split` `normal` `uniform` `randint` `categorical` |
| `mlx.fast` | `mlx.core.fast` | `rms-norm` `layer-norm` `rope` `scaled-dot-product-attention` `metal-kernel` |
| `mlx.distributed` | `mlx.core.distributed` | `init` `all-sum` `all-gather` `send` `recv` |
| `mlx.nn` | `mlx.nn` | `module` `linear` `conv2d` `multi-head-attention` `cross-entropy` `value-and-grad` |
| `mlx.optimizers` | `mlx.optimizers` | `adam` `adamw` `sgd` `cosine-decay` `clip-grad-norm` |
| `mlx.llm` (system `mlx/llm`) | `mlx-lm` | `load-model` `generate` `encode` `apply-chat-template` |
| `mlx.symreg` (system `mlx/symreg`) | (PySR, in spirit) | `symbolic-regression` `expression-function` `expression->mlx` |
| `mlx-ffi` | the C API, 1:1 | `mlx-array-new-data` `mlx-add` ... |

`mlx` **uses no packages**. Its symbols deliberately share names with CL
(`max`, `sum`, `abs`, `eval`, `load`, `compile`, ...), so use it through a
package-local nickname as shown above rather than `:use`-ing it. The
`mlx-user` package is set up that way for REPL work.

## Using it

**Arguments.** Every array argument also accepts Lisp data: numbers, Lisp
arrays, and nested lists/vectors. Lisp scalars follow MLX's weak typing, so
`(mx:add half-array 1.0)` stays `float16`. Required arguments are positional;
everything with a sensible default is a keyword, and every op takes `:stream`.

```lisp
(mx:sum a :axis 1 :keepdims t)        ; :axis nil / integer / list picks the C variant
(mx:transpose a :axes '(1 0))
(mx:full '(2 3) 7)                    ; int32 because 7 is an integer
(mx:arange 0 1 0.25)                  ; float32
(random:normal :shape '(3 3) :key (random:key 0))
(mx:conv2d x w :stride-0 2 :stride-1 2 :padding-0 1 :padding-1 1)
```

**Conversion.** `from-lisp` / `to-lisp` / `item`. `to-lisp` returns
specialized Lisp arrays (`single-float`, `(signed-byte 32)`, ...) by memcpy;
pass `:as :list` for nested lists. Specialized Lisp arrays are copied in one
step. Lists and T arrays go through type-declared loops, one per dtype, at
about 8 ns per element (2 ns for T arrays).

**Syntax.** CL-style n-ary arithmetic and chained comparisons, `@` for
matrix products, and optional `#M` array literals:

```lisp
(mx:+ a b 1)            ; a + b + 1
(mx:- a)                ; -a
(mx:/ a b 2)            ; a / b / 2
(mx:< 0 x 1)            ; 0 < x < 1, elementwise
(mx:@ q (mx:transpose k) v)

(setf *readtable* (copy-readtable))
(mx:enable-array-syntax)
#M((1 2) (3 4))         ; a 2x2 int32 array
#M:float16(0.5 1.5)     ; with a dtype
```

**Indexing.** `(mx:ref a 0 '(1 nil) t :newaxis)` is `a[0, 1:, :, None]`.
A list is always a slice `(start stop [step])`; gather with a vector or an
array, e.g. `(mx:ref a #(2 0))`. `(setf (mx:ref a t 1) 5)` updates `a`.

**Devices and streams.** Ops run on `mx:*stream*`, the default device's
stream unless bound:

```lisp
(mx:with-device (:cpu) (linalg:inv m))
(mx:add a b :stream :cpu)
(mx:with-stream ((mx:make-stream :gpu)) ...)
```

**Transformations.** `grad`, `value-and-grad`, `vjp`, `jvp`, `vmap`,
`compile`, `checkpoint`, `custom-function`, `custom-vjp`. Arguments may be
"pytrees": nested lists, plists, vectors and hash tables of arrays.

```lisp
(let* ((params (list :w (mx:zeros '(2 1)) :b (mx:zeros '(1))))
       (loss (lambda (p x y)
               (mx:mean (mx:square (mx:subtract (mx:add (mx:matmul x (getf p :w)) (getf p :b)) y)))))
       (step (mx:compile (mx:value-and-grad loss))))
  (multiple-value-bind (l grads) (funcall step params x y)   ; grads is a plist like params
    ...))
```

If a Lisp error is signalled inside a traced function, it propagates out of
the transform as the original condition.

**I/O.** `mx:load` dispatches on extension (`.npy`, `.safetensors`, `.gguf`).
Other entry points: `save`, `save-safetensors`, `save-gguf`, `gguf-metadata`,
and the in-memory variants `save-to-octets` / `load-from-octets` /
`save-safetensors-to-octets` / `load-safetensors-from-octets`.
`export-function` / `import-function` exchange traced graphs with MLX in
Python, C++ and Swift.

**Custom Metal kernels.**

```lisp
(let ((k (fast:metal-kernel :name "myexp" :input-names '("inp") :output-names '("out")
                            :source "uint i = thread_position_in_grid.x; out[i] = metal::exp(inp[i]);")))
  (funcall k :inputs (list x) :template '(("T" . :float32)) :grid '(1024) :threadgroup '(256)
             :output-shapes (list (mx:shape x)) :output-dtypes '(:float32)))
```

**Also included:** memory statistics and limits (`active-memory`,
`set-cache-limit`, ...), `export-to-dot` / `print-graph`, Metal capture,
compile modes, distributed groups, and device info.

## Neural networks

`mlx.nn` follows Python's `mlx.nn`: a module holds named children (arrays,
modules, or lists of them). Names follow the Python/Hugging Face convention
(`:q-proj` means `"q_proj"`), so weight files load by name. Modules are
funcallable.

```lisp
(nn:defmodule mlp () ())

(defun make-mlp (in hidden out)
  (let ((m (make-instance 'mlp)))
    (nn:register m :layers (list (nn:linear in hidden) (nn:linear hidden out)))
    m))

(defmethod nn:forward ((m mlp) &rest args)
  (destructuring-bind (l1 l2) (nn:child m :layers)
    (funcall l2 (nn:relu (funcall l1 (first args))))))

(let* ((model (make-mlp 784 256 10))
       (opt (optim:adamw 1e-3))
       (step (nn:value-and-grad model (lambda (x y)
                                        (nn:cross-entropy (funcall model x) y :reduction :mean)))))
  (dotimes (i 1000)
    (mx:with-scope ()
      (multiple-value-bind (loss grads) (funcall step x y)
        (optim:update opt model grads)
        (mx:eval (nn:parameters model) (optim:state opt))))))
```

**Layers:** `linear`, `embedding`, `conv1d`, `conv2d`, max/avg pooling,
`layer-norm`, `rms-norm`, `group-norm`, `batch-norm`, `dropout`,
`sequential`, `multi-head-attention`, `rope`, `prelu`, and the quantized
`quantized-linear` / `quantized-embedding` (see `nn:quantize`).

**Mixture-of-experts layers**, as in mlx-lm's `switch_layers`:
`switch-linear` (a stack of expert weights applied per token via
`gather_mm`), its quantized form (built by `nn:quantize`), and `switch-glu`.
When there are many tokens, `switch-glu` sorts them by expert first.

**Also:** 20 activations, 11 losses and the standard initializers. The rest
of the module protocol is `parameters`, `trainable-parameters`, `update`,
`freeze`, `train-mode`, `load-weights`, `save-weights` and `summary`.

**Optimizers:** `sgd`, `rmsprop`, `adagrad`, `adadelta`, `adam`, `adamw`,
`adamax` and `lion`. Schedules: `cosine-decay`, `exponential-decay`,
`step-decay`, `linear-schedule` and `join-schedules`. Also `clip-grad-norm`.

**Parity with Python MLX.** The test suite compares every activation, loss,
optimizer and schedule against values produced by Python MLX 0.32. As in
`mlx.nn`, activations run as shapeless-compiled (fused) kernels, which makes
them faster and their bf16 results identical to Python's.

`optim:update` frees the parameter and optimizer-state arrays it replaces,
so memory stays flat across training steps. Arrays stored in a module,
optimizer or cache are exempt from `with-scope` (see `mx:persist`).

### `defnet`: shapes checked at compile time

`nn:defnet` describes a network as a dataflow of stages. Its macro infers
every intermediate shape while it expands:

```lisp
(nn:defnet cnn ((x (batch 28 28 1)) &key (classes 10))
  (-> x
      (conv2d 16 3 :padding 1) relu (max-pool-2d 2)    ; (batch 14 14 16)
      (conv2d 32 3 :padding 1) relu (max-pool-2d 2)    ; (batch 7 7 32)
      flatten                                          ; (batch 1568)
      (linear 128) relu (dropout 0.25)                 ; linear gets its 1568 inputs itself
      (linear classes)))

(funcall (cnn :classes 3) images)    ; the constructor takes the hyperparameters
(nn:net-summary 'cnn)                ; each stage's shape and parameter count
```

- **Checks at compile time.** Layers take their input sizes from the
  inferred shapes. A mismatch signals `nn:shape-error` when the file
  compiles, naming the form:

  ```
  defnet lm: a residual branch must keep the shape (b s 64), but it gives (b s 32)
    in (residual (layer-norm) (linear 32))
  defnet bad: attention: 7 heads do not divide the width 60
  defnet bad: cannot broadcast (b 6) with (b 5): 6 vs 5
  ```

- **Three kinds of dimension.** A dimension can be:
  - an integer;
  - a hyperparameter (an `&key` parameter, known when the network is made);
  - a runtime dimension (any other symbol, such as `batch`), bound from the
    inputs on each call and checked for consistency across them.
- **Composition.** A one-input network can be a stage of another. Its
  hyperparameters are inferred from the shape it is applied to:

  ```lisp
  (nn:defnet decoder-block ((x (b s dims)) &key dims (heads 4) (hidden 256))
    (-> x
        (residual (layer-norm) (attention heads :mask :causal))
        (residual (layer-norm) (linear hidden) gelu (linear dims))))

  (nn:defnet lm ((tokens (b s)) &key (vocab 1000) (dims 64))
    (-> tokens (embedding vocab dims) (repeat 2 (decoder-block)) (rms-norm) (linear vocab)))
  ```

**Stages:**
- `linear`, `conv1d`, `conv2d`, max/avg pooling, `embedding`, `layer-norm`,
  `rms-norm`, `dropout` and `attention`;
- `flatten`, `reshape`, `transpose`, and `mean`/`sum`/`max` over an axis;
- the activations;
- `residual`, `repeat` and `elementwise`.

**Expressions:** `->`, `let*`, the elementwise operators `+ - * /
maximum minimum` (with broadcasting), `matmul` and `concat`. The MNIST
example's CNN is written this way.

## Language models

```lisp
(asdf:load-system "mlx/llm")

(let ((model (mlx.llm:load-model "HuggingFaceTB/SmolLM2-135M-Instruct")))
  (mlx.llm:generate model "What is the capital of France?"
                    :temperature 0.7 :stream *standard-output*))
```

`load-model` takes a local directory or a Hugging Face repo id. Repos are
downloaded on first use to `~/.cache/mlx-cl/models` (`$MLX_CL_CACHE`), and
`$HF_TOKEN` is sent for gated models. Supported model types:

- `llama`, `mistral`, `qwen2` and `qwen3`, which cover SmolLM, TinyLlama,
  Llama 3.x, Qwen2.5 and Qwen3 (`:thinking nil` / `--no-think` asks Qwen3
  to answer without reasoning);
- `phi3` (Phi-3 and 3.5, including LongRoPE);
- `gemma2` (attention and logit soft-capping);
- `gemma3_text` (sliding-window layers, q/k norms);
- mixture of experts: `mixtral`, `qwen2_moe` (with a shared expert),
  `qwen3_moe` and `olmoe`, each with its own routing rules. Raw Hugging
  Face checkpoints with per-expert weights are stacked on load.
- MLX 4- and 8-bit quantized checkpoints (e.g. from `mlx-community`).

On the CPU, MLX supports unquantized mixture-of-experts only in float32;
quantized MoE models work on both devices.

One configurable decoder implements them all. Each family's departures
from Llama follow mlx-lm operation for operation, down to details such as
Gemma 3 rounding its embedding scale to bfloat16 before a float16 multiply.

The BPE tokenizer is written from scratch and reads `tokenizer.json`. It
covers both families:

- byte-level (GPT-2, SmolLM, Llama 3, Qwen2);
- SentencePiece-style, with byte fallback (Gemma, Phi-3, Llama 2).

It matches the Hugging Face `tokenizers` library token for token on all six.

Chat templates are rendered directly rather than through a Jinja
interpreter. ChatML, Llama 3, Gemma and Phi-3 formats are supported, and
match `apply_chat_template` for every model above. Generation also stops at
the template's end-of-turn token, which some checkpoints omit from their
EOS list.

**Verified against mlx-lm** on an M3, with both on the same MLX version:

- Greedy generation gives identical tokens and text for SmolLM2-135M,
  Qwen2.5-0.5B, Qwen3-0.6B, Llama-3.2-1B, Gemma-2-2B, Gemma-3-1B,
  Phi-3.5-mini and OLMoE-1B-7B (4-bit mlx-community checkpoints).
- Mixtral, Qwen2-MoE and Qwen3-MoE are too large to run here. For them and
  every other new type, tiny random models built by mlx-lm give
  bit-identical logits (`tools/make-model-fixtures.py`, tested offline).
- Decoding speed matches mlx-lm's within measurement noise when the two run
  back to back, e.g. about 250 tokens/s for SmolLM2-135M and 54 for
  Gemma-2-2B.

To get there, the generation loop:

- queues step *n+1* on the GPU before blocking on token *n*;
- prefills the prompt exactly as mlx-lm does, since the schedule affects
  bf16 rounding;
- keeps a chunked, preallocated KV cache.

### Writing Lisp

`write-lisp` asks a model for code, runs it with tests, and feeds any
failure back until the tests pass:

```lisp
(mlx.llm:write-lisp (mlx.llm:load-model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
                    "Define (flatten tree) returning the atoms of a nested list in order"
                    :tests '("(equal (flatten '(1 (2 (3)) 4)) '(1 2 3 4))"))
```

```
$ bin/mlx-cl lisp -v 'Define (primes-below n) using a sieve' \
    -T "(equal (primes-below 20) '(2 3 5 7 11 13 17 19))"
```

It uses Lisp's structure at every step:

- **Reader-constrained decoding.** A lexer follows the reader through
  strings, `|symbols|`, character literals, comments and `#` dispatch. At each
  step, every token that would break the form is masked out. The model can
  only produce a prefix of exactly one readable form. `#.` (read-time
  evaluation) cannot be written at all.
- **Auto-close.** If the model tries to end its turn with forms still open,
  having lost count of its parens, the form is closed for it.
- **Parens from indentation.** Models indent Lisp well but miscount closing
  parens. When the two disagree, the closers are re-derived from the
  indentation, as in Parinfer's indent mode, and both readings are tried.
  This fixes the common failure where the closers run one short and later
  forms end up inside a `defun`.
- **A REPL loop.** Code runs in a fresh package, by default in a child
  SBCL with a timeout (`:isolation :in-process` uses a thread instead). The
  model gets concise feedback, clipped to keep the prompt small:
  - the error, including stack exhaustion;
  - compiler warnings;
  - each failing `(equal (f x) y)` test, as "`(f x)` returned *z*, expected *y*";
  - hints when it reaches for libraries that aren't there.

  Retries are sampled at temperature 0.7, since a greedy retry tends to
  resubmit the same code.

`generate-lisp-form` (constrained decoding alone) and `evaluate-lisp` are
also exported. Binding `mlx.llm::*constraint-trace*` to a stream reports
each token where the constraint overrode the model.

Qwen2.5-Coder-7B (4-bit, 4.5 GB peak, about 18 tokens/s on an M3) solves
typical exercises in one or two attempts, such as:

- `flatten`, run-length encoding and matrix multiplication;
- a `while` macro;
- a prime sieve.

Harder exercises can exhaust the attempts on logic errors. The 3B model is
noticeably weaker at Lisp; its output matches mlx-lm's, so the model is the
limit, not the implementation.

## Symbolic regression

`mlx/symreg` searches for a formula that fits data. Candidates are Lisp
expressions; genetic programming breeds them and the GPU scores them.

```lisp
(asdf:load-system "mlx/symreg")

(mlx.symreg:symbolic-regression rows ys :variables '(mass distance) :time-limit 60)
;; => #<CANDIDATE size 6 loss 3.442E-11 (* 6.674 (/ mass (square distance)))>, front
```

```
$ bin/mlx-cl symreg -f '(+ (square x0) (* 2.5 (sin x1)))'   # rediscover a formula
$ bin/mlx-cl symreg -t 60 gravity.csv                       # last column = target
...
size  loss (MSE / variance)  expression
   1               1.00      7.3012
   5              0.547      (+ 22.834 (* -5.1656 distance))
   6              3.442E-11  (* 6.674 (/ mass (square distance)))

best: (lambda (mass distance) (* 6.674003 (/ mass (square distance))))
```

The whole population is evaluated as one computation:

- **One compiled GPU program.** Each generation compiles every expression
  to a fixed-length postfix program. A vectorized stack machine, one MLX
  graph compiled once, runs all 1000 programs on all samples at the same
  time. Operators are protected (division by ~0, log and sqrt of
  negatives, exp overflow), so no candidate produces NaN, which would
  otherwise poison the gradients.
- **Constants tuned by gradient.** The same graph is differentiated with
  respect to the programs' constants, and Adam tunes every candidate's
  constants at once, inside the compiled step.
- **Linear scaling.** Each candidate f is judged as a + b·f with the
  least-squares a and b, computed in closed form on the GPU. Evolution then
  only has to find a formula's shape.
- **Evolution on the Lisp side.** Subtree crossover, subtree, point, hoist
  and constant mutations, tournament selection and elitism all work on
  plain s-expressions.

The result is a Pareto front of size against loss. `expression-function`
compiles any expression into a Lisp function, and `expression->mlx` builds
it as an MLX graph.

On an M3, a generation of 1000 candidates over 300 samples takes about
0.8 s. A test suite of six problems (Kepler's law, a harmonic mean, a
Gaussian, a cubic, a trig/rational mix and a double logarithm, three seeds
each) was solved to a loss below 1e-8 in 12 of 18 runs, with a 30-second
limit on each run.

## Memory management

Each MLX object is owned by a Lisp handle with a finalizer, so garbage
collection eventually frees it. MLX memory lives outside the Lisp heap,
though, and GC can't see that pressure. In loops, free deterministically:

```lisp
(dotimes (i 1000)
  (mx:with-scope ()                          ; frees everything created inside,
    (setf params (mx:keep (update params)))  ; except KEEP-ed values and the return value
    (mx:eval params)))
```

`mx:free` releases a handle immediately. Using a freed handle signals a Lisp
error; it never touches freed memory.

Handles created inside `with-scope` skip finalizer registration, which costs
more than a small op; only survivors leaving the outermost scope get one.

## Performance

Per-op overhead is about 20% above the raw C call. Inside `with-scope`, an
array op including its free costs roughly what an unmanaged op does.

Lisp scalars (`(mx:multiply g 0.01)`) are cached as shared constant arrays
per value and dtype. Without the cache, each one allocates a Metal buffer.
Calling `mx:free` inside a loop matters more than any of this: MLX memory
held until the GC runs costs far more than the Lisp wrapper does.

## CLI

```
$ bin/mlx-cl info                       # version, devices, memory
$ bin/mlx-cl eval '(mx:matmul (mx:eye 2) (mx:ones (list 2 3)))'
$ bin/mlx-cl eval --dot '(mx:exp (mx:ones (list 2)))' | dot -Tpng > graph.png
$ bin/mlx-cl bench -n 4096 -i 20        # matmul GFLOP/s; -d cpu for the CPU
$ bin/mlx-cl inspect model.safetensors  # tensor names, dtypes, shapes, sizes
$ bin/mlx-cl train -s 200               # compiled value-and-grad linear regression
$ bin/mlx-cl generate "Write a haiku about Lisp"            # SmolLM2-135M by default
$ bin/mlx-cl generate -m mlx-community/Qwen2.5-0.5B-Instruct-4bit -t 0 -v "Explain monads"
$ bin/mlx-cl chat -m mlx-community/gemma-3-1b-it-4bit
$ bin/mlx-cl generate -m mlx-community/Qwen3-0.6B-4bit --no-think "Explain monads"
$ bin/mlx-cl download mlx-community/Qwen2.5-0.5B-Instruct-4bit
$ bin/mlx-cl lisp -v 'Define (flatten tree)' -T "(equal (flatten '(1 (2))) '(1 2))"
$ bin/mlx-cl symreg -f '(* x0 (sqrt x0))'                    # symbolic regression
```

## Implementation notes

- **No libffi, no C shim.** Every mlx-c handle is `struct { void* ctx; }`.
  Under AAPCS64 that is passed and returned exactly like a pointer, so
  handles are `:pointer` aliases. `mlx_optional_*` structs (8 bytes, not
  HFAs) travel in one register and are packed into a `uint64`. The 72-byte
  `mlx_io_vtable` is passed by reference, as the ABI requires for large
  composites. The two-word map iterators return `map_ctx` in `x1`; we pass
  the map pointer back instead, since disassembly shows the two are always
  the same.
- **Errors.** mlx-c's default error handler calls `exit()`. We install a
  handler that records the message, and every call's status code is turned
  into a `mlx:mlx-error` condition.
- **Callbacks.** Lisp functions become MLX closures through a payload
  registry and static trampolines. A Lisp error inside a callback is caught
  before it can unwind through C++ frames, reported to MLX as a failure, and
  re-signalled afterwards. MLX may call I/O callbacks from its own threads;
  SBCL supports this.
- **Float traps.** Metal's allocator and Accelerate leave IEEE exception
  flags set, and SBCL on arm64 turns them into Lisp errors after a foreign
  call returns. Every `mlx-ffi` function is therefore an inline wrapper that
  masks traps, cheaply when they are already masked, so the raw layer is
  safe to call directly. High-level ops mask once for all their calls.
- **Saved images.** `sb-ext:*init-hooks*` reinstalls the error handler and
  drops cached streams, so `bin/mlx-cl` starts cleanly.
- The CUDA kernel API is bound in `mlx-ffi` only (no CUDA on macOS).
