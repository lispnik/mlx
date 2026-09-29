;;;; nn/layers.lisp -- standard layers
;;;;
;;;; Each layer is a module class plus a constructor function of the same
;;;; name: (nn:linear 784 128) makes an instance of class NN:LINEAR.
;;;; Parameter names and layouts match mlx.nn (and therefore the weight
;;;; files written by MLX and Hugging Face).

(in-package :mlx.nn.impl)

(defun pair (x) (if (listp x) x (list x x)))

(defun uniform-init (shape scale &optional (dtype :float32))
  (random:uniform :shape shape :low (- scale) :high scale :dtype dtype))

;;; Linear

(nn:defmodule nn:linear ()
  ((input-dims :initarg :input-dims)
   (output-dims :initarg :output-dims)))

(defun nn:linear (input-dims output-dims &key (bias t))
  "y = x W^T + b, with W of shape (OUTPUT-DIMS INPUT-DIMS)."
  (let ((m (make-instance 'nn:linear :input-dims input-dims :output-dims output-dims))
        (scale (sqrt (/ 1.0 input-dims))))
    (nn:register m "weight" (uniform-init (list output-dims input-dims) scale))
    (when bias (nn:register m "bias" (uniform-init (list output-dims) scale)))
    m))

(defmethod nn:forward ((m nn:linear) &rest args)
  (destructuring-bind (x) args
    (let ((w (mx:transpose (nn:child m "weight")))
          (b (nn:child m "bias")))
      ;; as mlx.nn.Linear: a fused addmm, so the bias is added before rounding
      (if b (mx:addmm b x w) (mx:matmul x w)))))

(defmethod module-description ((m nn:linear))
  (with-slots (input-dims output-dims) m
    (format nil "~D -> ~D~:[~;, bias~]" input-dims output-dims (nn:child m "bias"))))

;;; Embedding

(nn:defmodule nn:embedding ()
  ((num-embeddings :initarg :num-embeddings)
   (dims :initarg :dims)))

(defun nn:embedding (num-embeddings dims)
  "A lookup table of NUM-EMBEDDINGS vectors of size DIMS."
  (let ((m (make-instance 'nn:embedding :num-embeddings num-embeddings :dims dims)))
    (nn:register m "weight" (random:normal :shape (list num-embeddings dims)
                                           :scale (sqrt (/ 1.0 dims))))
    m))

(defmethod nn:forward ((m nn:embedding) &rest args)
  (destructuring-bind (indices) args
    (mx:take (nn:child m "weight") indices :axis 0)))

(defgeneric nn:as-linear (module x)
  (:documentation "Use an embedding's weight as an output projection (tied weights).")
  (:method ((m nn:embedding) x)
    (mx:matmul x (mx:transpose (nn:child m "weight")))))

(defmethod module-description ((m nn:embedding))
  (with-slots (num-embeddings dims) m (format nil "~D x ~D" num-embeddings dims)))

;;; Normalization

(nn:defmodule nn:layer-norm ()
  ((dims :initarg :dims) (eps :initarg :eps)))

(defun nn:layer-norm (dims &key (eps 1e-5) (affine t) (bias t))
  (let ((m (make-instance 'nn:layer-norm :dims dims :eps eps)))
    (when affine
      (nn:register m "weight" (mx:ones (list dims)))
      (when bias (nn:register m "bias" (mx:zeros (list dims)))))
    m))

(defmethod nn:forward ((m nn:layer-norm) &rest args)
  (destructuring-bind (x) args
    (fast:layer-norm x :weight (nn:child m "weight") :bias (nn:child m "bias")
                       :eps (slot-value m 'eps))))

(nn:defmodule nn:rms-norm ()
  ((dims :initarg :dims) (eps :initarg :eps)))

(defun nn:rms-norm (dims &key (eps 1e-5))
  (let ((m (make-instance 'nn:rms-norm :dims dims :eps eps)))
    (nn:register m "weight" (mx:ones (list dims)))
    m))

(defmethod nn:forward ((m nn:rms-norm) &rest args)
  (destructuring-bind (x) args
    (fast:rms-norm x :weight (nn:child m "weight") :eps (slot-value m 'eps))))

(nn:defmodule nn:group-norm ()
  ((groups :initarg :groups) (dims :initarg :dims) (eps :initarg :eps)))

(defun nn:group-norm (groups dims &key (eps 1e-5) (affine t))
  "Normalize over groups of the feature (last) axis."
  (let ((m (make-instance 'nn:group-norm :groups groups :dims dims :eps eps)))
    (when affine
      (nn:register m "weight" (mx:ones (list dims)))
      (nn:register m "bias" (mx:zeros (list dims))))
    m))

(defmethod nn:forward ((m nn:group-norm) &rest args)
  (destructuring-bind (x) args
    (with-slots (groups dims eps) m
      (let* ((shape (mx:shape x))
             (batch (first shape))
             (grouped (mx:reshape x (list batch -1 groups (/ dims groups))))
             (grouped (mx:transpose grouped :axes '(0 2 1 3)))
             (grouped (mx:reshape grouped (list batch groups -1)))
             (normed (fast:layer-norm grouped :eps eps))
             (normed (mx:reshape normed (list batch groups -1 (/ dims groups))))
             (y (mx:reshape (mx:transpose normed :axes '(0 2 1 3)) shape)))
        (if (nn:child m "weight")
            (mx:add (mx:multiply y (nn:child m "weight")) (nn:child m "bias"))
            y)))))

(nn:defmodule nn:batch-norm ()
  ((dims :initarg :dims) (eps :initarg :eps) (momentum :initarg :momentum)))

(defun nn:batch-norm (dims &key (eps 1e-5) (momentum 0.1) (affine t) (track-running-stats t))
  "Batch normalization over all axes but the last.  Running statistics are
frozen children, updated in training mode and used in evaluation mode."
  (let ((m (make-instance 'nn:batch-norm :dims dims :eps eps :momentum momentum)))
    (when affine
      (nn:register m "weight" (mx:ones (list dims)))
      (nn:register m "bias" (mx:zeros (list dims))))
    (when track-running-stats
      (nn:register m "running_mean" (mx:zeros (list dims)))
      (nn:register m "running_var" (mx:ones (list dims)))
      (nn:freeze m :keys '("running_mean" "running_var") :recurse nil))
    m))

(defmethod nn:forward ((m nn:batch-norm) &rest args)
  (destructuring-bind (x) args
    (with-slots (eps momentum) m
      (let* ((axes (loop for i below (1- (mx:ndim x)) collect i))
             (tracking (nn:child m "running_mean"))
             (use-batch (or (nn:training-p m) (not tracking)))
             (mean (if use-batch (mx:mean x :axis axes) (nn:child m "running_mean")))
             (var (if use-batch (mx:var x :axis axes) (nn:child m "running_var"))))
        (when (and tracking (nn:training-p m))
          (flet ((blend (old new) (mx:add (mx:multiply old (- 1 momentum)) (mx:multiply new momentum))))
            (nn:register m "running_mean" (blend (nn:child m "running_mean") (mx:stop-gradient mean)))
            (nn:register m "running_var" (blend (nn:child m "running_var") (mx:stop-gradient var)))))
        (let ((y (mx:multiply (mx:subtract x mean) (mx:rsqrt (mx:add var eps)))))
          (if (nn:child m "weight")
              (mx:add (mx:multiply y (nn:child m "weight")) (nn:child m "bias"))
              y))))))

;;; Dropout

(nn:defmodule nn:dropout () ((p :initarg :p)))

(defun nn:dropout (&optional (p 0.5))
  "Zero elements with probability P in training mode (scaling the rest)."
  (unless (<= 0 p 1) (error "Dropout probability must be in [0, 1], not ~A." p))
  (make-instance 'nn:dropout :p p))

(defmethod nn:forward ((m nn:dropout) &rest args)
  (destructuring-bind (x) args
    (let ((p (slot-value m 'p)))
      (if (or (not (nn:training-p m)) (zerop p))
          x
          (let ((keep (random:bernoulli :p (- 1 p) :shape (mx:shape x))))
            (mx:multiply (mx:where keep x 0) (/ 1 (- 1 p))))))))

;;; Convolution (channels last: NLC / NHWC)

(nn:defmodule nn:conv1d ()
  ((stride :initarg :stride) (padding :initarg :padding)
   (dilation :initarg :dilation) (groups :initarg :groups)))

(defun nn:conv1d (in-channels out-channels kernel-size
                  &key (stride 1) (padding 0) (dilation 1) (groups 1) (bias t))
  (let ((m (make-instance 'nn:conv1d :stride stride :padding padding :dilation dilation :groups groups))
        (scale (sqrt (/ 1.0 (* (/ in-channels groups) kernel-size)))))
    (nn:register m "weight" (uniform-init (list out-channels kernel-size (/ in-channels groups)) scale))
    (when bias (nn:register m "bias" (mx:zeros (list out-channels))))
    m))

(defmethod nn:forward ((m nn:conv1d) &rest args)
  (destructuring-bind (x) args
    (with-slots (stride padding dilation groups) m
      (let ((y (mx:conv1d x (nn:child m "weight") :stride stride :padding padding
                                                   :dilation dilation :groups groups)))
        (if (nn:child m "bias") (mx:add y (nn:child m "bias")) y)))))

(nn:defmodule nn:conv2d ()
  ((stride :initarg :stride) (padding :initarg :padding)
   (dilation :initarg :dilation) (groups :initarg :groups)))

(defun nn:conv2d (in-channels out-channels kernel-size
                  &key (stride 1) (padding 0) (dilation 1) (groups 1) (bias t))
  "KERNEL-SIZE, STRIDE, PADDING and DILATION are integers or (h w) pairs."
  (let* ((kernel (pair kernel-size))
         (m (make-instance 'nn:conv2d :stride (pair stride) :padding (pair padding)
                                      :dilation (pair dilation) :groups groups))
         (scale (sqrt (/ 1.0 (* (/ in-channels groups) (first kernel) (second kernel))))))
    (nn:register m "weight" (uniform-init (list out-channels (first kernel) (second kernel)
                                                (/ in-channels groups))
                                          scale))
    (when bias (nn:register m "bias" (mx:zeros (list out-channels))))
    m))

(defmethod nn:forward ((m nn:conv2d) &rest args)
  (destructuring-bind (x) args
    (with-slots (stride padding dilation groups) m
      (let ((y (mx:conv2d x (nn:child m "weight")
                          :stride-0 (first stride) :stride-1 (second stride)
                          :padding-0 (first padding) :padding-1 (second padding)
                          :dilation-0 (first dilation) :dilation-1 (second dilation)
                          :groups groups)))
        (if (nn:child m "bias") (mx:add y (nn:child m "bias")) y)))))

;;; Pooling: sliding windows via as-strided, then a reduction over them

(defun sliding-windows (x window strides)
  "View X (N, spatial..., C) as (N, out-spatial..., window..., C)."
  (let* ((shape (mx:shape x))
         (spatial (subseq shape 1 (1- (length shape))))
         ;; row-major element strides of X
         (x-strides (let ((acc 1) (out '()))
                      (dolist (d (reverse shape) out) (push acc out) (setf acc (* acc d)))))
         (spatial-strides (subseq x-strides 1 (1- (length x-strides))))
         (out-shape (append (list (first shape))
                            (mapcar (lambda (size w s) (1+ (floor (- size w) s))) spatial window strides)
                            window
                            (last shape)))
         (out-strides (append (list (first x-strides))
                              (mapcar #'* spatial-strides strides)
                              spatial-strides
                              (last x-strides))))
    (mx:as-strided x out-shape out-strides)))

(nn:defmodule pool ()
  ((kernel :initarg :kernel) (stride :initarg :stride) (padding :initarg :padding)
   (reducer :initarg :reducer) (pad-value :initarg :pad-value)))

(defmethod nn:forward ((m pool) &rest args)
  (destructuring-bind (x) args
    (with-slots (kernel stride padding reducer pad-value) m
      (let* ((x (if (some #'plusp padding)
                    (mx:pad x (append '((0 0)) (mapcar (lambda (p) (list p p)) padding) '((0 0)))
                            :constant-values pad-value)
                    x))
             (windows (sliding-windows x kernel stride))
             (n (length kernel))
             (window-axes (loop for i from (1+ n) repeat n collect i)))
        (funcall reducer windows window-axes)))))

(defun make-pool (dims kernel-size stride padding reducer pad-value)
  (flet ((expand (v) (if (listp v) v (make-list dims :initial-element v))))
    (make-instance 'pool :kernel (expand kernel-size) :stride (expand (or stride kernel-size))
                         :padding (expand padding) :reducer reducer :pad-value pad-value)))

(defun max-reducer (w axes) (mx:max w :axis axes))
(defun mean-reducer (w axes) (mx:mean w :axis axes))

(defun nn:max-pool-1d (kernel-size &key stride (padding 0))
  (make-pool 1 kernel-size stride padding #'max-reducer sb-ext:single-float-negative-infinity))
(defun nn:max-pool-2d (kernel-size &key stride (padding 0))
  (make-pool 2 kernel-size stride padding #'max-reducer sb-ext:single-float-negative-infinity))
(defun nn:avg-pool-1d (kernel-size &key stride (padding 0))
  (make-pool 1 kernel-size stride padding #'mean-reducer 0))
(defun nn:avg-pool-2d (kernel-size &key stride (padding 0))
  (make-pool 2 kernel-size stride padding #'mean-reducer 0))

;;; Containers and simple modules

(nn:defmodule nn:sequential () ())

(defun nn:sequential (&rest layers)
  "Apply LAYERS (modules or functions) in order."
  (let ((m (make-instance 'nn:sequential)))
    (nn:register m "layers" layers)
    m))

(defmethod module-description ((m nn:sequential))
  (format nil "~D layers" (length (nn:child m "layers"))))

(defmethod nn:forward ((m nn:sequential) &rest args)
  (destructuring-bind (x) args
    (reduce (lambda (acc layer) (funcall layer acc)) (nn:child m "layers") :initial-value x)))

(nn:defmodule nn:identity () ())
(defun nn:identity () (make-instance 'nn:identity))
(defmethod nn:forward ((m nn:identity) &rest args) (first args))

(nn:defmodule nn:prelu () ())

(defun nn:prelu (&key (num-parameters 1) (init 0.25))
  (let ((m (make-instance 'nn:prelu)))
    (nn:register m "weight" (mx:full (list num-parameters) init :dtype :float32))
    m))

(defmethod nn:forward ((m nn:prelu) &rest args)
  (destructuring-bind (x) args
    (mx:add (mx:maximum x 0) (mx:multiply (nn:child m "weight") (mx:minimum x 0)))))

;;; Attention

(defun nn:create-additive-causal-mask (n &key (dtype :float32))
  "An (N N) mask with 0 on and below the diagonal and a large negative
number above it, to add to attention scores."
  (let ((indices (mx:arange n)))
    (mx:multiply (mx:astype (mx:less (mx:expand-dims indices 1) (mx:expand-dims indices 0)) dtype)
                 -1e9)))

(nn:defmodule nn:multi-head-attention () ((num-heads :initarg :num-heads)))

(defun nn:multi-head-attention (dims num-heads &key query-input-dims key-input-dims
                                                     value-input-dims value-dims value-output-dims
                                                     (bias nil))
  (unless (zerop (mod dims num-heads))
    (error "DIMS (~D) must be divisible by NUM-HEADS (~D)." dims num-heads))
  (let ((m (make-instance 'nn:multi-head-attention :num-heads num-heads))
        (query-input-dims (or query-input-dims dims))
        (key-input-dims (or key-input-dims dims)))
    (nn:register m "query_proj" (nn:linear query-input-dims dims :bias bias))
    (nn:register m "key_proj" (nn:linear key-input-dims dims :bias bias))
    (nn:register m "value_proj" (nn:linear (or value-input-dims key-input-dims) (or value-dims dims)
                                           :bias bias))
    (nn:register m "out_proj" (nn:linear (or value-dims dims) (or value-output-dims dims) :bias bias))
    m))

(defun split-heads (x num-heads)
  "(B L D) -> (B heads L D/heads)"
  (destructuring-bind (b l d) (mx:shape x)
    (mx:transpose (mx:reshape x (list b l num-heads (/ d num-heads))) :axes '(0 2 1 3))))

(defmethod nn:forward ((m nn:multi-head-attention) &rest args)
  (destructuring-bind (queries keys values &key mask) args
    (let* ((h (slot-value m 'num-heads))
           (q (split-heads (funcall (nn:child m "query_proj") queries) h))
           (k (split-heads (funcall (nn:child m "key_proj") keys) h))
           (v (split-heads (funcall (nn:child m "value_proj") values) h))
           (scale (/ 1.0 (sqrt (car (last (mx:shape q))))))
           (out (etypecase mask
                  (null (fast:scaled-dot-product-attention q k v scale))
                  ((member :causal) (fast:scaled-dot-product-attention q k v scale :mask-mode "causal"))
                  (mx:mlx-array (fast:scaled-dot-product-attention q k v scale :mask-mode "array"
                                                                               :mask-arr mask))))
           (out (mx:transpose out :axes '(0 2 1 3))))
      (destructuring-bind (b l hh d) (mx:shape out)
        (funcall (nn:child m "out_proj") (mx:reshape out (list b l (* hh d))))))))

(nn:defmodule nn:rope ()
  ((dims :initarg :dims) (traditional :initarg :traditional)
   (base :initarg :base) (scale :initarg :scale)))

(defun nn:rope (dims &key traditional (base 10000.0) (scale 1.0))
  "Rotary positional encoding of the first DIMS features."
  (make-instance 'nn:rope :dims dims :traditional traditional :base base :scale scale))

(defmethod nn:forward ((m nn:rope) &rest args)
  (destructuring-bind (x &key (offset 0)) args
    (with-slots (dims traditional base scale) m
      (fast:rope x dims :traditional traditional :base (float base 1.0) :scale scale
                        :offset offset))))

;;; Quantized layers

(nn:defmodule nn:quantized-linear ()
  ((group-size :initarg :group-size) (bits :initarg :bits) (mode :initarg :mode)))

(defun quantize-into (m weight group-size bits mode)
  (destructuring-bind (wq scales &optional biases) (mx:quantize weight :group-size group-size
                                                                       :bits bits :mode mode)
    (nn:register m "weight" wq)
    (nn:register m "scales" scales)
    (when biases (nn:register m "biases" biases))
    (nn:freeze m :keys '("weight" "scales" "biases") :recurse nil)))

(defun nn:quantized-linear (input-dims output-dims &key (bias t) (group-size 64) (bits 4)
                                                        (mode "affine") from)
  "A linear layer with a quantized weight.  FROM, a LINEAR, supplies the
weights (else they are random).  The quantized weights are frozen."
  (let ((m (make-instance 'nn:quantized-linear :group-size group-size :bits bits :mode mode))
        (source (or from (nn:linear input-dims output-dims :bias bias))))
    (quantize-into m (nn:child source "weight") group-size bits mode)
    (when (nn:child source "bias") (nn:register m "bias" (nn:child source "bias")))
    m))

(defmethod nn:forward ((m nn:quantized-linear) &rest args)
  (destructuring-bind (x) args
    (with-slots (group-size bits mode) m
      (let ((y (mx:quantized-matmul x (nn:child m "weight") (nn:child m "scales")
                                    :biases (nn:child m "biases") :transpose t
                                    :group-size group-size :bits bits :mode mode)))
        (if (nn:child m "bias") (mx:add y (nn:child m "bias")) y)))))

(nn:defmodule nn:quantized-embedding ()
  ((group-size :initarg :group-size) (bits :initarg :bits) (mode :initarg :mode)))

(defun nn:quantized-embedding (num-embeddings dims &key (group-size 64) (bits 4) (mode "affine") from)
  (let ((m (make-instance 'nn:quantized-embedding :group-size group-size :bits bits :mode mode)))
    (quantize-into m (nn:child (or from (nn:embedding num-embeddings dims)) "weight")
                   group-size bits mode)
    m))

(defmethod nn:forward ((m nn:quantized-embedding) &rest args)
  (destructuring-bind (indices) args
    (with-slots (group-size bits mode) m
      (let ((rows (lambda (name) (and (nn:child m name) (mx:take (nn:child m name) indices :axis 0)))))
        (mx:dequantize (funcall rows "weight") (funcall rows "scales")
                       :biases (funcall rows "biases")
                       :group-size group-size :bits bits :mode mode)))))

(defmethod nn:as-linear ((m nn:quantized-embedding) x)
  (with-slots (group-size bits mode) m
    (mx:quantized-matmul x (nn:child m "weight") (nn:child m "scales")
                         :biases (nn:child m "biases") :transpose t
                         :group-size group-size :bits bits :mode mode)))

;;; Mixture-of-experts layers (mlx-lm's switch_layers): a stack of expert
;;; weights, applied per token to the experts chosen by a router

(nn:defmodule nn:switch-linear () ())

(defun nn:switch-linear (input-dims output-dims num-experts &key (bias t))
  "NUM-EXPERTS linear layers in one (experts out in) weight.  Called with
(x indices &key sorted): each row of X goes through the experts INDICES names."
  (let ((m (make-instance 'nn:switch-linear))
        (scale (sqrt (/ 1.0 input-dims))))
    (nn:register m "weight" (uniform-init (list num-experts output-dims input-dims) scale))
    (when bias (nn:register m "bias" (mx:zeros (list num-experts output-dims))))
    m))

(defun add-expert-bias (m x indices)
  (let ((b (nn:child m "bias")))
    (if b (mx:add x (mx:expand-dims (mx:take b indices :axis 0) -2)) x)))

(defmethod nn:forward ((m nn:switch-linear) &rest args)
  (destructuring-bind (x indices &key sorted) args
    (add-expert-bias m (mx:gather-mm x (mx:swapaxes (nn:child m "weight") -1 -2)
                                     :rhs-indices indices :sorted-indices sorted)
                     indices)))

(nn:defmodule nn:quantized-switch-linear ()
  ((group-size :initarg :group-size) (bits :initarg :bits) (mode :initarg :mode)))

(defun nn:quantized-switch-linear (&key from (group-size 64) (bits 4) (mode "affine"))
  "A SWITCH-LINEAR with quantized expert weights, made FROM a switch-linear."
  (let ((m (make-instance 'nn:quantized-switch-linear :group-size group-size :bits bits :mode mode)))
    (quantize-into m (nn:child from "weight") group-size bits mode)
    (when (nn:child from "bias") (nn:register m "bias" (nn:child from "bias")))
    m))

(defmethod nn:forward ((m nn:quantized-switch-linear) &rest args)
  (destructuring-bind (x indices &key sorted) args
    (with-slots (group-size bits mode) m
      (add-expert-bias m (mx:gather-qmm x (nn:child m "weight") (nn:child m "scales")
                                        :biases (nn:child m "biases") :rhs-indices indices
                                        :transpose t :group-size group-size :bits bits :mode mode
                                        :sorted-indices sorted)
                       indices))))

(defun nn:swiglu (gate x)
  "silu(GATE) * X as one fused kernel, as mlx-lm computes it."
  (funcall (compiled 'nn:swiglu (lambda (gate x) (mx:multiply (mx:multiply gate (mx:sigmoid gate)) x)))
           gate x))

(nn:defmodule nn:switch-glu () ())

(defun nn:switch-glu (input-dims hidden-dims num-experts &key bias)
  "A SwiGLU MLP per expert.  Called with (x indices), INDICES (..., k)
naming each token's experts; returns (..., k, input-dims)."
  (let ((m (make-instance 'nn:switch-glu)))
    (nn:register m "gate_proj" (nn:switch-linear input-dims hidden-dims num-experts :bias bias))
    (nn:register m "up_proj" (nn:switch-linear input-dims hidden-dims num-experts :bias bias))
    (nn:register m "down_proj" (nn:switch-linear hidden-dims input-dims num-experts :bias bias))
    m))

(defmethod nn:forward ((m nn:switch-glu) &rest args)
  (destructuring-bind (x indices) args
    (let* ((x (mx:expand-dims x '(-2 -3)))
           ;; with many tokens, group them by expert so weights are read in order
           (sort (>= (mx:size indices) 64))
           (k (mx:dim indices -1))
           (flat (and sort (mx:flatten indices)))
           (order (and sort (mx:argsort flat)))
           (inverse (and sort (mx:argsort order)))
           (x (if sort (mx:take (mx:flatten x :start-axis 0 :end-axis -3) (mx:floor-divide order k) :axis 0) x))
           (idx (if sort (mx:take flat order :axis 0) indices))
           (idx (if (nn:training-p m) (mx:stop-gradient idx) idx))
           (up (funcall (nn:child m "up_proj") x idx :sorted sort))
           (gate (funcall (nn:child m "gate_proj") x idx :sorted sort))
           (y (funcall (nn:child m "down_proj") (nn:swiglu gate up) idx :sorted sort))
           (y (if sort (mx:unflatten (mx:take y inverse :axis 0) 0 (mx:shape indices)) y)))
      (mx:squeeze y :axis -2))))

(defun nn:quantize (model &key (group-size 64) (bits 4) (mode "affine") (predicate (constantly t)))
  "Replace MODEL's LINEAR, EMBEDDING and SWITCH-LINEAR layers (for which
PREDICATE, called with (path module), is true) by quantized versions, in place.  Layers whose
input size is not a multiple of GROUP-SIZE are left alone.  Returns MODEL."
  (let ((replacements '()))
    (nn:apply-to-modules
     (lambda (path m)
       (let ((w (and (typep m '(or nn:linear nn:embedding nn:switch-linear)) (nn:child m "weight"))))
         (when (and w (zerop (mod (car (last (mx:shape w))) group-size))
                    (funcall predicate path m))
           (push (cons path (etypecase m
                              (nn:linear (nn:quantized-linear 0 0 :group-size group-size :bits bits
                                                                  :mode mode :from m))
                              (nn:embedding (nn:quantized-embedding 0 0 :group-size group-size
                                                                        :bits bits :mode mode :from m))
                              (nn:switch-linear (nn:quantized-switch-linear :from m :group-size group-size
                                                                            :bits bits :mode mode))))
                 replacements))))
     model)
    (loop for (path . new) in replacements
          do (multiple-value-bind (container key)
                 (locate-slot model (uiop:split-string path :separator "."))
               (if (typep container 'nn:module)
                   (setf (nn:child container key) new)
                   (setf (car container) new))))
    model))
