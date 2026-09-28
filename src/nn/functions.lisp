;;;; nn/functions.lisp -- activations, losses and initializers
;;;;
;;;; These follow mlx.nn's definitions and defaults (including each loss's
;;;; default REDUCTION), so Python code ports over directly.

(in-package :mlx.nn.impl)

;;; ------------------------------------------------------------------
;;; Activations
;;;
;;; As in mlx.nn, activations run as shapeless-compiled (fused) functions:
;;; one kernel, one rounding to the input dtype -- faster, and numerically
;;; identical to Python MLX.  Called with non-default parameters they run
;;; unfused.

(defvar *compiled* (make-hash-table :test 'eq)
  "Name -> compiled function, created on first use in each process.")

(defun compiled (name function)
  (or (gethash name *compiled*)
      (setf (gethash name *compiled*) (mx:compile function :shapeless t))))

(defun forget-compiled () (clrhash *compiled*))
(pushnew 'forget-compiled sb-ext:*init-hooks*)

(defmacro define-activation (name (x &rest keys) documentation body)
  "Define activation NAME of X and keyword parameters KEYS ((key default)...).
BODY computes it; it is compiled when every key has its default value."
  (let ((raw (intern (format nil "%~A" (symbol-name name)))))
    `(progn
       (defun ,raw (,x ,@(mapcar #'first keys))
         (declare (ignorable ,@(mapcar #'first keys)))
         ,body)
       (defun ,name (,x ,@(when keys `(&key ,@keys)))
         ,documentation
         (if (and ,@(loop for (k d) in keys collect `(eql ,k ,d)))
             (funcall (compiled ',name (lambda (,x) (,raw ,x ,@(mapcar #'second keys)))) ,x)
             (,raw ,x ,@(mapcar #'first keys)))))))

(defun nn:tanh (x) (mx:tanh x))

(define-activation nn:sigmoid (x) "1 / (1 + exp(-x))." (mx:sigmoid x))
(define-activation nn:relu (x) "max(x, 0)." (mx:maximum x 0))
(define-activation nn:relu6 (x) "min(max(x, 0), 6)." (mx:minimum (mx:maximum x 0) 6))
(define-activation nn:leaky-relu (x (negative-slope 0.01)) "max(slope * x, x)."
  (mx:maximum (mx:multiply x negative-slope) x))

(define-activation nn:elu (x (alpha 1.0)) "x if x > 0, else alpha * (exp(x) - 1)."
  (mx:where (mx:greater x 0) x (mx:multiply (mx:expm1 x) alpha)))

(define-activation nn:selu (x) "Scaled ELU."
  (mx:multiply (%elu x 1.67326319217681884765625) 1.05070102214813232421875))

(define-activation nn:celu (x (alpha 1.0)) "Continuously differentiable ELU."
  (mx:add (mx:maximum x 0)
          (mx:multiply (mx:expm1 (mx:divide (mx:minimum x 0) alpha)) alpha)))

(define-activation nn:gelu (x) "Exact GELU: x * Phi(x)."
  (mx:divide (mx:multiply x (mx:add 1 (mx:erf (mx:divide x (sqrt 2.0))))) 2))

(define-activation nn:gelu-approx (x) "GELU with the tanh approximation."
  (mx:multiply (mx:multiply x 0.5)
               (mx:add 1 (mx:tanh (mx:multiply (sqrt (/ 2 pi))
                                               (mx:add x (mx:multiply (mx:power x 3) 0.044715)))))))

(define-activation nn:gelu-fast-approx (x) "GELU as x * sigmoid(1.702 x)."
  (mx:multiply x (mx:sigmoid (mx:multiply x 1.702))))

(define-activation nn:silu (x) "x * sigmoid(x)." (mx:multiply x (mx:sigmoid x)))
(define-activation nn:softplus (x) "log(1 + exp(x))." (mx:logaddexp x 0))
(define-activation nn:mish (x) "x * tanh(softplus(x))." (mx:multiply x (mx:tanh (%softplus x))))
(define-activation nn:softsign (x) "x / (1 + |x|)." (mx:divide x (mx:add 1 (mx:abs x))))
(define-activation nn:log-sigmoid (x) "log(sigmoid(x))." (mx:negative (%softplus (mx:negative x))))
(define-activation nn:hardswish (x) "x * relu6(x + 3) / 6."
  (mx:divide (mx:multiply x (mx:minimum (mx:maximum (mx:add x 3) 0) 6)) 6))
(define-activation nn:hard-tanh (x (min-val -1.0) (max-val 1.0)) "clip(x, min-val, max-val)."
  (mx:clip x :a-min min-val :a-max max-val))
(define-activation nn:softmax (x (axis -1)) "Softmax along AXIS." (mx:softmax x :axis axis))
(define-activation nn:log-softmax (x (axis -1)) "Log of softmax along AXIS."
  (mx:subtract x (mx:logsumexp x :axis axis :keepdims t)))
(define-activation nn:step (x (threshold 0.0)) "1 where X >= THRESHOLD, else 0."
  (mx:where (mx:greater-equal x threshold) 1 0))

(defun nn:glu (x &key (axis -1))
  "Gated linear unit: split X in two along AXIS, a * sigmoid(b)."
  (destructuring-bind (a b) (mx:split x 2 :axis axis)
    (mx:multiply a (mx:sigmoid b))))

;;; ------------------------------------------------------------------
;;; Losses

(defun reduce-loss (loss reduction)
  (ecase reduction
    (:none loss)
    (:mean (mx:mean loss))
    (:sum (mx:sum loss))))

(defun weighted (loss weights)
  (if weights (mx:multiply loss weights) loss))

(defun nn:cross-entropy (logits targets &key weights (axis -1) (label-smoothing 0.0) (reduction :none))
  "Cross entropy between unnormalized LOGITS and TARGETS, which are class
indices (one dimension less than LOGITS) or probabilities (same shape)."
  (let* ((targets (mx:ensure-array targets))
         (probabilities-p (= (mx:ndim targets) (mx:ndim logits)))
         (score (if probabilities-p
                    (mx:sum (mx:multiply logits targets) :axis axis)
                    (mx:squeeze (mx:take-along-axis logits (mx:expand-dims targets -1) axis)
                                :axis -1)))
         (score (if (plusp label-smoothing)
                    (mx:add (mx:multiply score (- 1 label-smoothing))
                            (mx:multiply (mx:mean logits :axis axis) label-smoothing))
                    score))
         (loss (mx:subtract (mx:logsumexp logits :axis axis) score)))
    (reduce-loss (weighted loss weights) reduction)))

(defun nn:binary-cross-entropy (inputs targets &key weights (with-logits t) (reduction :mean))
  "Binary cross entropy.  INPUTS are logits, or probabilities when
WITH-LOGITS is NIL."
  (let ((loss (if with-logits
                  (mx:subtract (mx:logaddexp 0 inputs) (mx:multiply inputs targets))
                  (let ((eps 1e-12))
                    (mx:negative
                     (mx:add (mx:multiply targets (mx:log (mx:maximum inputs eps)))
                             (mx:multiply (mx:subtract 1 targets)
                                          (mx:log (mx:maximum (mx:subtract 1 inputs) eps)))))))))
    (reduce-loss (weighted loss weights) reduction)))

(defun nn:nll-loss (inputs targets &key (axis -1) (reduction :none))
  "Negative log likelihood of class indices TARGETS under log-probabilities INPUTS."
  (reduce-loss (mx:negative (mx:squeeze (mx:take-along-axis inputs (mx:expand-dims targets -1) axis)
                                        :axis -1))
               reduction))

(defun nn:mse-loss (predictions targets &key (reduction :mean))
  (reduce-loss (mx:square (mx:subtract predictions targets)) reduction))

(defun nn:l1-loss (predictions targets &key (reduction :mean))
  (reduce-loss (mx:abs (mx:subtract predictions targets)) reduction))

(defun nn:smooth-l1-loss (predictions targets &key (beta 1.0) (reduction :mean))
  (let ((diff (mx:abs (mx:subtract predictions targets))))
    (reduce-loss (mx:where (mx:less diff beta)
                           (mx:divide (mx:multiply (mx:square diff) 0.5) beta)
                           (mx:subtract diff (* 0.5 beta)))
                 reduction)))

(defun nn:huber-loss (inputs targets &key (delta 1.0) (reduction :none))
  (let* ((errors (mx:abs (mx:subtract inputs targets)))
         (quadratic (mx:minimum errors delta))
         (linear (mx:subtract errors quadratic)))
    (reduce-loss (mx:add (mx:multiply (mx:square quadratic) 0.5) (mx:multiply linear delta))
                 reduction)))

(defun nn:kl-div-loss (inputs targets &key (axis -1) (reduction :none))
  "KL divergence; INPUTS and TARGETS are log-probabilities."
  (reduce-loss (mx:sum (mx:multiply (mx:exp targets) (mx:subtract targets inputs)) :axis axis)
               reduction))

(defun nn:log-cosh-loss (inputs targets &key (reduction :none))
  (let ((errors (mx:subtract inputs targets)))
    (reduce-loss (mx:subtract (mx:logaddexp errors (mx:negative errors)) (log 2.0)) reduction)))

(defun nn:cosine-similarity-loss (x1 x2 &key (axis 1) (eps 1e-8) (reduction :none))
  (flet ((norm (x) (mx:sqrt (mx:sum (mx:square x) :axis axis))))
    (reduce-loss (mx:divide (mx:sum (mx:multiply x1 x2) :axis axis)
                            (mx:maximum (mx:multiply (norm x1) (norm x2)) eps))
                 reduction)))

(defun nn:hinge-loss (inputs targets &key (reduction :none))
  (reduce-loss (mx:maximum (mx:subtract 1 (mx:multiply inputs targets)) 0) reduction))

;;; ------------------------------------------------------------------
;;; Initializers
;;;
;;; Each returns a function taking an array and returning a new array of
;;; the same shape (and DTYPE), e.g.
;;;   (nn:update model (mx:tree-map (nn:glorot-uniform) (nn:parameters model)))

(defun nn:init-constant (value &key (dtype :float32))
  (lambda (a) (mx:full (mx:shape a) value :dtype dtype)))

(defun nn:init-normal (&key (mean 0.0) (std 1.0) (dtype :float32))
  (lambda (a) (random:normal :shape (mx:shape a) :dtype dtype :loc mean :scale std)))

(defun nn:init-uniform (&key (low 0.0) (high 1.0) (dtype :float32))
  (lambda (a) (random:uniform :shape (mx:shape a) :dtype dtype :low low :high high)))

(defun nn:init-identity (&key (dtype :float32))
  (lambda (a)
    (destructuring-bind (rows cols) (mx:shape a)
      (unless (= rows cols) (error "INIT-IDENTITY needs a square matrix, not ~S." (mx:shape a)))
      (mx:eye rows :dtype dtype))))

(defun fans (shape)
  "(values fan-in fan-out) for a weight of SHAPE (out, ..., in)."
  (if (< (length shape) 2)
      (values (first shape) (first shape))
      (let ((receptive (reduce #'* (subseq shape 1 (1- (length shape))))))
        (values (* (car (last shape)) receptive) (* (first shape) receptive)))))

(defun nn:glorot-normal (&key (dtype :float32))
  (lambda (a &key (gain 1.0))
    (multiple-value-bind (in out) (fans (mx:shape a))
      (random:normal :shape (mx:shape a) :dtype dtype
                     :scale (* gain (sqrt (/ 2.0 (+ in out))))))))

(defun nn:glorot-uniform (&key (dtype :float32))
  (lambda (a &key (gain 1.0))
    (multiple-value-bind (in out) (fans (mx:shape a))
      (let ((limit (* gain (sqrt (/ 6.0 (+ in out))))))
        (random:uniform :shape (mx:shape a) :dtype dtype :low (- limit) :high limit)))))

(defun nn:he-normal (&key (dtype :float32))
  (lambda (a &key (mode :fan-in) (gain 1.0))
    (multiple-value-bind (in out) (fans (mx:shape a))
      (random:normal :shape (mx:shape a) :dtype dtype
                     :scale (/ gain (sqrt (ecase mode (:fan-in in) (:fan-out out))))))))

(defun nn:he-uniform (&key (dtype :float32))
  (lambda (a &key (mode :fan-in) (gain 1.0))
    (multiple-value-bind (in out) (fans (mx:shape a))
      (let ((limit (* gain (sqrt (/ 3.0 (ecase mode (:fan-in in) (:fan-out out)))))))
        (random:uniform :shape (mx:shape a) :dtype dtype :low (- limit) :high limit)))))
