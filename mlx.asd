;;;; mlx.asd

(defsystem "mlx"
  :description "Comprehensive Common Lisp bindings to Apple MLX via mlx-c."
  :author "mkennedy"
  :license "MIT"
  :version "0.1.0"
  :depends-on ("cffi" "trivial-garbage" "uiop")
  :pathname "src/"
  :serial t
  :components ((:static-file "generated/exports.sexp")
               (:file "package" :depends-on ("generated/exports.sexp"))
               (:file "ffi/library")
               (:file "ffi/bindings")
               (:file "errors")
               (:file "handle")
               (:file "dtype")
               (:file "device")
               (:file "array")
               (:file "vector")
               (:file "op")
               (:file "generated/ops")
               (:file "ops-extra")
               (:file "tree")
               (:file "closure")
               (:file "transforms")
               (:file "io")
               (:file "export")
               (:file "kernels")
               (:file "system")
               (:file "init")
               (:file "syntax")
               (:file "nn/module")
               (:file "nn/functions")
               (:file "nn/layers")
               (:file "nn/optimizers"))
  :in-order-to ((test-op (test-op "mlx/tests"))))

(defsystem "mlx/tests"
  :description "FiveAM test suite for mlx."
  :depends-on ("mlx" "fiveam")
  :pathname "tests/"
  :serial t
  :components ((:file "package")
               (:file "array")
               (:file "ops")
               (:file "linalg-fft-random")
               (:file "transforms")
               (:file "io")
               (:file "system")
               (:file "nn"))
  :perform (test-op (o c) (symbol-call :mlx-tests :run-tests)))

(defsystem "mlx/llm"
  :description "Language models on MLX: Hugging Face tokenizers, Llama-family models, generation."
  :depends-on ("mlx" "com.inuoe.jzon")
  :pathname "src/llm/"
  :serial t
  :components ((:file "package")
               (:file "tokenizer")
               (:file "models")
               (:file "hub")
               (:file "generate")
               (:file "lisp")))

(defsystem "mlx/symreg"
  :description "Symbolic regression: genetic programming over Lisp expressions, evaluated and tuned on the GPU."
  :depends-on ("mlx")
  :pathname "src/symreg/"
  :serial t
  :components ((:file "package")
               (:file "symreg")))

(defsystem "mlx/symreg-tests"
  :description "FiveAM tests for mlx/symreg."
  :depends-on ("mlx/symreg" "fiveam")
  :pathname "tests/"
  :components ((:file "symreg"))
  :perform (test-op (o c) (symbol-call :mlx-symreg-tests :run-tests)))

(defsystem "mlx/llm-tests"
  :description "FiveAM tests for mlx/llm."
  :depends-on ("mlx/llm" "fiveam")
  :pathname "tests/"
  :components ((:static-file "fixtures/pretokenize.sexp")
               (:static-file "fixtures/smollm.sexp")
               (:static-file "fixtures/families.sexp")
               (:module "fixtures/tiny" :components ((:static-file "MLX_VERSION")))
               (:file "llm"))
  :perform (test-op (o c) (symbol-call :mlx-llm-tests :run-tests)))

(defsystem "mlx/cli"
  :description "Command-line driver for mlx, built with clingon."
  :depends-on ("mlx" "mlx/llm" "mlx/symreg" "clingon")
  :pathname "cli/"
  :serial t
  :components ((:file "main"))
  :build-operation "program-op"
  :build-pathname "../bin/mlx-cl"
  :entry-point "mlx-cli:main")
