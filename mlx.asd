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
               (:file "init"))
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
               (:file "system"))
  :perform (test-op (o c) (symbol-call :mlx-tests :run-tests)))

(defsystem "mlx/cli"
  :description "Command-line driver for mlx, built with clingon."
  :depends-on ("mlx" "clingon")
  :pathname "cli/"
  :serial t
  :components ((:file "main"))
  :build-operation "program-op"
  :build-pathname "../bin/mlx-cl"
  :entry-point "mlx-cli:main")
