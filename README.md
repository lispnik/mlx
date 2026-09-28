# mlx — Common Lisp bindings to Apple MLX

[![CI](https://github.com/lispnik/mlx/actions/workflows/ci.yml/badge.svg)](https://github.com/lispnik/mlx/actions/workflows/ci.yml)

Complete SBCL bindings to [mlx-c](https://github.com/ml-explore/mlx-c), the C API
of Apple's MLX array framework: lazy n-dimensional arrays on the Apple Silicon GPU
and CPU, automatic differentiation, vectorization, graph compilation, custom
Metal kernels, and safetensors/GGUF I/O.

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
make test              # FiveAM suite: 471 checks (MLX_CL_TEST_DEVICE=cpu forces the CPU)
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
| `tests/` | FiveAM suite |
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
pass `:as :list` for nested lists.

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
