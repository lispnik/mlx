;;;; package.lisp -- package definitions
;;;;
;;;; MLX-FFI       raw 1:1 bindings (generated; exports every C function)
;;;; MLX           the public API.  It uses no packages, so MLX:MAX, MLX:SUM,
;;;;               MLX:EVAL... are distinct from CL's.  Refer to it with a
;;;;               package-local nickname, e.g. (:local-nicknames (:mx :mlx)).
;;;; MLX.LINALG, MLX.FFT, MLX.RANDOM, MLX.FAST, MLX.DISTRIBUTED
;;;;               sub-namespaces mirroring mlx.core.<name> in Python.
;;;; MLX.IMPL      implementation package; uses CL, so MAX there is CL:MAX.
;;;; MLX-USER      a convenience package for the REPL.

(defpackage :mlx-ffi
  (:use :cl)
  (:documentation "Raw CFFI bindings to mlx-c, generated from its headers.
Every function is exported under its lispified C name, e.g. MLX-ARRAY-NEW-DATA."))

(defpackage :mlx.build
  (:use :cl)
  (:export #:exports))

(in-package :mlx.build)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun exports (package-name &rest handwritten)
    "Export list for PACKAGE-NAME: generated names from generated/exports.sexp
plus HANDWRITTEN symbol names."
    (let* ((here (or *compile-file-truename* *load-truename*))
           (file (merge-pathnames "generated/exports.sexp" here))
           (generated (with-open-file (in file) (let ((*read-eval* nil)) (read in)))))
      (cons :export
            (remove-duplicates
             (append (cdr (assoc package-name generated :test #'string=))
                     (mapcar #'string handwritten))
             :test #'string=)))))

(in-package :cl-user)

(defpackage :mlx
  (:use)
  (:documentation "Common Lisp interface to Apple MLX (via mlx-c).
This package uses no other package: its symbols deliberately share names
with CL (MAX, SUM, EVAL, LOAD, COMPILE...).  Use a package-local nickname.")
  #.(mlx.build:exports
     "MLX"
     ;; types and conditions
     '#:mlx-array '#:mlx-array-p '#:mlx-stream '#:mlx-stream-p '#:mlx-device '#:mlx-device-p
     '#:mlx-error '#:mlx-error-message
     ;; lifetime
     '#:free '#:freed-p '#:with-scope '#:keep '#:persist
     ;; construction and conversion
     '#:from-lisp '#:to-lisp '#:item '#:scalar '#:ensure-array '#:dtypes
     '#:arange '#:pad '#:split '#:tensordot
     ;; properties
     '#:shape '#:ndim '#:size '#:dtype '#:itemsize '#:nbytes '#:strides '#:dim '#:dtype-size
     ;; indexing
     '#:ref
     ;; evaluation
     '#:eval '#:async-eval
     ;; devices and streams
     '#:*stream* '#:with-stream '#:with-device '#:default-device '#:set-default-device
     '#:make-device '#:device-type '#:device-index '#:device-info '#:device-count
     '#:device-available-p '#:default-stream '#:set-default-stream '#:make-stream
     '#:stream-device '#:stream-index '#:synchronize
     ;; transforms
     '#:grad '#:value-and-grad '#:vjp '#:jvp '#:vmap '#:compile '#:checkpoint
     '#:custom-vjp '#:custom-function
     '#:enable-compile '#:disable-compile '#:set-compile-mode '#:clear-compile-cache
     ;; trees
     '#:tree-flatten '#:tree-unflatten '#:tree-map
     ;; io
     '#:load '#:save '#:load-safetensors '#:save-safetensors '#:load-gguf '#:save-gguf
     '#:gguf-metadata '#:save-to-octets '#:load-from-octets
     '#:save-safetensors-to-octets '#:load-safetensors-from-octets
     ;; function export
     '#:export-function '#:import-function '#:with-function-exporter
     ;; system
     '#:version '#:metal-available-p '#:cuda-available-p
     '#:start-metal-capture '#:stop-metal-capture
     '#:active-memory '#:cache-memory '#:peak-memory '#:reset-peak-memory
     '#:memory-limit '#:set-memory-limit '#:set-cache-limit '#:set-wired-limit '#:clear-cache
     '#:export-to-dot '#:print-graph
     ;; syntax
     '#:+ '#:- '#:* '#:/ '#:@ '#:< '#:> '#:<= '#:>= '#:= '#:/=
     '#:enable-array-syntax '#:disable-array-syntax
     '#:reinitialize))

(defpackage :mlx.linalg
  (:use)
  (:documentation "Linear algebra (mlx.core.linalg).")
  #.(mlx.build:exports "MLX.LINALG" '#:norm))

(defpackage :mlx.fft
  (:use)
  (:documentation "Fast Fourier transforms (mlx.core.fft).")
  #.(mlx.build:exports "MLX.FFT"))

(defpackage :mlx.random
  (:use)
  (:documentation "Random number generation (mlx.core.random).")
  #.(mlx.build:exports "MLX.RANDOM" '#:seed '#:split '#:categorical '#:permutation))

(defpackage :mlx.fast
  (:use)
  (:documentation "Fused fast operations and custom Metal kernels (mlx.core.fast).")
  #.(mlx.build:exports "MLX.FAST" '#:metal-kernel))

(defpackage :mlx.distributed
  (:use)
  (:documentation "Distributed communication (mlx.core.distributed).")
  #.(mlx.build:exports "MLX.DISTRIBUTED"
                       '#:available-p '#:init '#:group-rank '#:group-size '#:group-split))

(defpackage :mlx.nn
  (:use)
  (:documentation "Neural networks (mlx.nn): modules, layers, activations,
losses and initializers.")
  (:export
   ;; module protocol
   #:module #:defmodule #:forward #:child #:children #:register #:modules
   #:parameters #:trainable-parameters #:update #:flatten-parameters #:parameter-count
   #:freeze #:unfreeze #:train-mode #:training-p #:apply-to-modules
   #:load-weights #:save-weights #:value-and-grad #:summary
   ;; layers
   #:linear #:embedding #:as-linear #:layer-norm #:rms-norm #:group-norm #:batch-norm
   #:dropout #:conv1d #:conv2d #:max-pool-1d #:max-pool-2d #:avg-pool-1d #:avg-pool-2d
   #:sequential #:identity #:prelu #:multi-head-attention #:create-additive-causal-mask
   #:rope #:quantized-linear #:quantized-embedding #:quantize
   #:switch-linear #:quantized-switch-linear #:switch-glu #:swiglu
   ;; activations
   #:relu #:relu6 #:leaky-relu #:elu #:selu #:celu #:gelu #:gelu-approx #:gelu-fast-approx
   #:silu #:mish #:softplus #:softsign #:log-sigmoid #:hardswish #:hard-tanh
   #:sigmoid #:tanh #:softmax #:log-softmax #:glu #:step
   ;; losses
   #:cross-entropy #:binary-cross-entropy #:nll-loss #:mse-loss #:l1-loss #:smooth-l1-loss
   #:huber-loss #:kl-div-loss #:log-cosh-loss #:cosine-similarity-loss #:hinge-loss
   ;; initializers
   #:init-constant #:init-normal #:init-uniform #:init-identity
   #:glorot-normal #:glorot-uniform #:he-normal #:he-uniform))

(defpackage :mlx.optimizers
  (:use)
  (:documentation "Optimizers, learning-rate schedules and gradient clipping (mlx.optimizers).")
  (:export
   #:optimizer #:update #:apply-gradients #:learning-rate #:state #:step-count #:reset
   #:sgd #:rmsprop #:adagrad #:adadelta #:adam #:adamw #:adamax #:lion
   #:exponential-decay #:step-decay #:cosine-decay #:linear-schedule #:join-schedules
   #:clip-grad-norm))

(defpackage :mlx.impl
  (:use :cl)
  (:local-nicknames (:ffi :mlx-ffi) (:tg :trivial-garbage)))

(defpackage :mlx.nn.impl
  (:use :cl)
  (:local-nicknames (:mx :mlx) (:nn :mlx.nn) (:optim :mlx.optimizers)
                    (:random :mlx.random) (:fast :mlx.fast) (:impl :mlx.impl)))

(defpackage :mlx-user
  (:use :cl)
  (:local-nicknames (:mx :mlx) (:linalg :mlx.linalg) (:fft :mlx.fft)
                    (:random :mlx.random) (:fast :mlx.fast)
                    (:nn :mlx.nn) (:optim :mlx.optimizers)))
