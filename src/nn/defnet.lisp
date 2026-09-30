;;;; nn/defnet.lisp -- DEFNET: networks whose shapes are checked at compile time
;;;;
;;;;   (nn:defnet classifier ((x (batch 28 28 1)) &key (classes 10))
;;;;     (-> x
;;;;         (conv2d 16 3 :padding 1) relu (max-pool-2d 2)    ; (batch 14 14 16)
;;;;         (conv2d 32 3 :padding 1) relu (max-pool-2d 2)    ; (batch 7 7 32)
;;;;         flatten                                          ; (batch 1568)
;;;;         (linear 128) relu (dropout 0.25)
;;;;         (linear classes)))                               ; (batch classes)
;;;;
;;;; The macro infers the shape of every intermediate value while it expands.
;;;; Layers take their input sizes from those shapes, so (linear 128) above
;;;; gets 1568 inputs.  Any mismatch -- a residual that changes shape, a
;;;; matmul whose inner dimensions differ, attention heads that don't divide
;;;; the width -- signals NN:SHAPE-ERROR at compile time, naming the form.
;;;;
;;;; Dimensions are integers, hyperparameters (the &key parameters: known
;;;; when the network is made), or runtime dimensions (other symbols, like
;;;; BATCH: bound from the inputs at each call, and checked for consistency
;;;; then).  DEFNET defines a module class, a constructor of the same name
;;;; taking the hyperparameters, and FORWARD.  A network with one input can
;;;; be a stage of another; its hyperparameters are inferred from the shape
;;;; it is applied to.
;;;;
;;;; Expressions:  input or LET* variable | number | (-> expr stage...)
;;;;   | (let* ((var expr)...) expr) | (+ - * / maximum minimum expr...)
;;;;   | (matmul a b) | (concat axis expr...)
;;;;   The body may return several values: (values expr...), also as the
;;;;   body of a LET* (e.g. heads sharing a trunk).
;;;; Stages:  (linear n) (conv1d out k) (conv2d out k) (conv-transpose1d out k)
;;;;   (conv-transpose2d out k) (max-pool-1d k) (max-pool-2d k) (avg-pool-1d k)
;;;;   (avg-pool-2d k) (embedding vocab dims) (layer-norm) (rms-norm)
;;;;   (batch-norm) (group-norm groups) (dropout p) (attention heads [:mask :causal])
;;;;   flatten (reshape dim...) (transpose axis...) (mean axis) (sum axis)
;;;;   (max axis) (residual stage...) (repeat n stage...) (elementwise fn)
;;;;   relu gelu silu tanh sigmoid softmax ... (activations, with keywords)
;;;;   (net keyword-args...) for a network defined with DEFNET

(in-package :mlx.nn.impl)

(define-condition nn:shape-error (error)
  ((net :initarg :net :initform nil :reader shape-error-net)
   (form :initarg :form :initform nil :reader shape-error-form)
   (message :initarg :message :reader shape-error-message))
  (:report (lambda (c s)
             (let ((*print-case* :downcase) (*print-pretty* nil))
               (format s "~@[defnet ~S: ~]~A~@[~%  in ~S~]"
                       (shape-error-net c) (shape-error-message c) (shape-error-form c))))))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defvar *nets* (make-hash-table :test 'eq)
    "DEFNET signatures by name, known at compile time: a plist of :INPUTS
(specs), :OUTPUT (shape), :HYPER ((name default)...) and :TRACE."))

;;; Expansion state
(defvar *net* nil)          ; the network being defined
(defvar *hyper* '())        ; its hyperparameter names
(defvar *self* nil)         ; the variable holding the module in FORWARD
(defvar *layers* '())       ; (child-name . constructor-form), reversed
(defvar *counts* nil)       ; child names used, by kind
(defvar *trace* '())        ; (description shape child-name-or-nil), reversed
(defvar *form* nil)         ; the form being checked, for messages

(defun shape-error (control &rest args)
  (error 'nn:shape-error :net *net* :form *form*
                         :message (let ((*print-case* :downcase) (*print-pretty* nil))
                                    (apply #'format nil control args))))

(defun name= (symbol name) (and (symbolp symbol) (string= (symbol-name symbol) name)))

;;; ------------------------------------------------------------------
;;; Dimensions: integers, symbols, and forms of them over + * floor

(defun static-dim-p (d)
  "D is known when the network is made (no runtime dimension in it)."
  (cond ((integerp d) t)
        ((symbolp d) (member d *hyper*))
        ((consp d) (every #'static-dim-p (rest d)))))

(defun dim-sum (args)
  "The sum of ARGS as a dimension: integers added, like terms collected."
  (let ((n 0) (terms '()))
    (labels ((add (d k)
               (cond ((integerp d) (incf n (* k d)))
                     ((and (consp d) (eq (first d) '+)) (dolist (x (rest d)) (add x k)))
                     ((and (consp d) (eq (first d) '*) (integerp (second d)))
                      (add (dim-product (cddr d)) (* k (second d))))
                     (t (let ((cell (assoc d terms :test #'equal)))
                          (if cell (incf (cdr cell) k) (push (cons d k) terms)))))))
      (dolist (a args) (add a 1)))
    (let ((parts (append (loop for (d . k) in (reverse terms)
                               unless (zerop k) collect (if (= k 1) d (dim-product (list k d))))
                         (unless (zerop n) (list n)))))
      (cond ((null parts) 0)
            ((null (rest parts)) (first parts))
            (t (cons '+ parts))))))

(defun dim-op (op &rest args)
  (when (eq op '+) (return-from dim-op (dim-sum args)))
  (let ((args (remove-if (lambda (a) (and (eql a 1) (eq op '*))) args)))
    (cond ((every #'integerp args) (if (eq op 'floor) (floor (first args) (second args)) (apply op args)))
          ((and (member op '(+ *)) (null (rest args))) (first args))
          (t (cons op args)))))

(defun dim-product (dims)
  "The product of DIMS as a dimension, with the integers multiplied out."
  (let ((n (reduce #'* (remove-if-not #'integerp dims)))
        (others (remove-if #'integerp dims)))
    (cond ((null others) n)
          ((and (= n 1) (null (rest others))) (first others))
          ((= n 1) (cons '* others))
          (t (list* '* n others)))))

(defun factors (dims)
  "DIMS as (integer . sorted list of other factors), for comparing products."
  (let ((n 1) (others '()))
    (labels ((add (d) (cond ((integerp d) (setf n (* n d)))
                            ((and (consp d) (eq (first d) '*)) (mapc #'add (rest d)))
                            (t (push d others)))))
      (mapc #'add dims))
    (cons n (sort others #'string< :key #'prin1-to-string))))

(defun window-output (size kernel stride padding)
  "Output length of a convolution or pooling window over SIZE."
  (let ((span (dim-op '+ size (- (* 2 padding) kernel))))
    (when (and (integerp span) (minusp span))
      (shape-error "a window of ~D does not fit an input of ~D (padding ~D)" kernel size padding))
    (if (= stride 1) (dim-op '+ span 1) (dim-op '+ (dim-op 'floor span stride) 1))))

(defun broadcast (a b)
  (let ((ra (reverse a)) (rb (reverse b)))
    (reverse
     (loop for i below (max (length a) (length b))
           collect (let ((x (nth i ra)) (y (nth i rb)))
                     (cond ((null x) y) ((null y) x)
                           ((equal x y) x) ((eql x 1) y) ((eql y 1) x)
                           (t (shape-error "cannot broadcast ~S with ~S: ~S vs ~S" a b x y))))))))

(defun normalize-axis (axis rank)
  (let ((a (if (minusp axis) (+ axis rank) axis)))
    (unless (< -1 a rank) (shape-error "axis ~D out of range for rank ~D" axis rank))
    a))

(defun require-rank (shape rank what)
  (unless (= (length shape) rank)
    (shape-error "~A needs a rank-~D input, got ~S" what rank shape)))

(defun static-width (shape what)
  (let ((d (car (last shape))))
    (unless shape (shape-error "~A needs an array, got a scalar" what))
    (unless (static-dim-p d)
      (shape-error "~A needs its input width known when the network is made, but it is ~S (in ~S)~
                    ~:[~;; make ~S a &key hyperparameter~]"
                   what d shape (symbolp d) d))
    d))

;;; ------------------------------------------------------------------
;;; Children and the trace

(defun add-layer (kind constructor)
  (let* ((n (incf (getf *counts* (intern (string kind) :keyword) -1)))
         (name (format nil "~(~A~)_~D" (substitute #\_ #\- (string kind)) n)))
    (push (cons name constructor) *layers*)
    name))

(defun call-layer (name code &rest args)
  `(funcall (nn:child ,*self* ,name) ,code ,@args))

(defun note (description shape &optional child)
  (push (list description shape child) *trace*)
  shape)

;;; ------------------------------------------------------------------
;;; Stages

(defvar *stages* (make-hash-table :test 'equal))

(defmacro define-stage (name (args shape code) &body body)
  "Define stage NAME: BODY gets the stage's ARGS, the incoming SHAPE and
the CODE computing the incoming value, and returns (values shape code)."
  `(setf (gethash ,(string name) *stages*)
         (lambda (,args ,shape ,code) (declare (ignorable ,args ,shape ,code)) ,@body)))

(defun layer-stage (kind shape code constructor out-shape &rest call-args)
  (declare (ignore shape))
  (let ((name (add-layer kind constructor)))
    (values (note (format nil "~(~A~)" kind) out-shape name)
            (apply #'call-layer name code call-args))))

(define-stage linear (args shape code)
  (destructuring-bind (n &key (bias t)) args
    (let ((in (static-width shape "linear")))
      (layer-stage 'linear shape code `(nn:linear ,in ,n :bias ,bias) (append (butlast shape) (list n))))))

(defun pair-of (v) (if (consp v) v (list v v)))

(define-stage conv1d (args shape code)
  (destructuring-bind (out k &key (stride 1) (padding 0)) args
    (require-rank shape 3 "conv1d")
    (let ((in (static-width shape "conv1d")))
      (layer-stage 'conv1d shape code `(nn:conv1d ,in ,out ,k :stride ,stride :padding ,padding)
                   (list (first shape) (window-output (second shape) k stride padding) out)))))

(define-stage conv2d (args shape code)
  (destructuring-bind (out k &key (stride 1) (padding 0)) args
    (require-rank shape 4 "conv2d")
    (let ((in (static-width shape "conv2d")))
      (destructuring-bind ((kh kw) (sh sw) (ph pw)) (mapcar #'pair-of (list k stride padding))
        (layer-stage 'conv2d shape code
                     `(nn:conv2d ,in ,out ',(list kh kw) :stride ',(list sh sw) :padding ',(list ph pw))
                     (list (first shape) (window-output (second shape) kh sh ph)
                           (window-output (third shape) kw sw pw) out))))))

(defun transposed-output (size kernel stride padding output-padding)
  "Output length of a transposed convolution over SIZE."
  (dim-op '+ (dim-op '* (dim-op '+ size -1) stride) (+ (- (* 2 padding)) (1- kernel) output-padding 1)))

(define-stage conv-transpose1d (args shape code)
  (destructuring-bind (out k &key (stride 1) (padding 0) (output-padding 0)) args
    (require-rank shape 3 "conv-transpose1d")
    (let ((in (static-width shape "conv-transpose1d")))
      (layer-stage 'conv-transpose1d shape code
                   `(nn:conv-transpose1d ,in ,out ,k :stride ,stride :padding ,padding :output-padding ,output-padding)
                   (list (first shape) (transposed-output (second shape) k stride padding output-padding) out)))))

(define-stage conv-transpose2d (args shape code)
  (destructuring-bind (out k &key (stride 1) (padding 0) (output-padding 0)) args
    (require-rank shape 4 "conv-transpose2d")
    (let ((in (static-width shape "conv-transpose2d")))
      (destructuring-bind ((kh kw) (sh sw) (ph pw) (oh ow)) (mapcar #'pair-of (list k stride padding output-padding))
        (layer-stage 'conv-transpose2d shape code
                     `(nn:conv-transpose2d ,in ,out ',(list kh kw) :stride ',(list sh sw) :padding ',(list ph pw)
                                           :output-padding ',(list oh ow))
                     (list (first shape) (transposed-output (second shape) kh sh ph oh)
                           (transposed-output (third shape) kw sw pw ow) out))))))

(macrolet ((pool (name maker dims)
             `(define-stage ,name (args shape code)
                (destructuring-bind (k &key stride (padding 0)) args
                  (require-rank shape ,(+ dims 2) ,(string-downcase name))
                  (let ((stride (or stride k)))
                    (layer-stage ',name shape code (list ',maker k :stride stride :padding padding)
                                 (append (list (first shape))
                                         (loop for d in (subseq shape 1 ,(1+ dims))
                                               collect (window-output d k stride padding))
                                         (last shape))))))))
  (pool max-pool-1d nn:max-pool-1d 1)
  (pool max-pool-2d nn:max-pool-2d 2)
  (pool avg-pool-1d nn:avg-pool-1d 1)
  (pool avg-pool-2d nn:avg-pool-2d 2))

(define-stage dropout (args shape code)
  (destructuring-bind (&optional (p 0.5)) args
    (layer-stage 'dropout shape code `(nn:dropout ,p) shape)))

(define-stage layer-norm (args shape code)
  (layer-stage 'layer-norm shape code `(nn:layer-norm ,(static-width shape "layer-norm") ,@args) shape))

(define-stage batch-norm (args shape code)
  (layer-stage 'batch-norm shape code `(nn:batch-norm ,(static-width shape "batch-norm") ,@args) shape))

(define-stage group-norm (args shape code)
  (destructuring-bind (groups &rest options) args
    (let ((width (static-width shape "group-norm")))
      (when (and (integerp width) (integerp groups) (plusp (mod width groups)))
        (shape-error "group-norm: ~D groups do not divide the width ~D" groups width))
      (layer-stage 'group-norm shape code `(nn:group-norm ,groups ,width ,@options) shape))))

(define-stage rms-norm (args shape code)
  (layer-stage 'rms-norm shape code `(nn:rms-norm ,(static-width shape "rms-norm") ,@args) shape))

(define-stage embedding (args shape code)
  (destructuring-bind (vocab dims) args
    (layer-stage 'embedding shape code `(nn:embedding ,vocab ,dims) (append shape (list dims)))))

(define-stage attention (args shape code)
  (destructuring-bind (heads &key mask) args
    (require-rank shape 3 "attention")
    (let ((d (static-width shape "attention")))
      (when (and (integerp d) (integerp heads) (plusp (mod d heads)))
        (shape-error "attention: ~D heads do not divide the width ~D" heads d))
      (let ((name (add-layer 'attention `(nn:multi-head-attention ,d ,heads)))
            (v (gensym "X")))
        (values (note (format nil "attention (~D heads)" heads) shape name)
                `(let ((,v ,code)) ,(call-layer name v v v :mask mask)))))))

(define-stage flatten (args shape code)
  (when (< (length shape) 2) (shape-error "flatten needs rank 2 or more, got ~S" shape))
  (let ((v (gensym "X")))
    (values (note "flatten" (list (first shape) (dim-product (rest shape))))
            `(let ((,v ,code)) (mx:reshape ,v (list (mx:dim ,v 0) -1))))))

(define-stage reshape (dims shape code)
  (let ((known (remove -1 dims)))
    (when (> (count -1 dims) 1) (shape-error "reshape takes at most one -1: ~S" dims))
    (destructuring-bind (in-n . in-f) (factors shape)
      (destructuring-bind (out-n . out-f) (factors known)
        (let ((rest-f (let ((f (copy-list in-f)))
                        (dolist (x out-f f)
                          (if (member x f :test #'equal)
                              (setf f (remove x f :test #'equal :count 1))
                              (shape-error "cannot reshape ~S to ~S: ~S is not a factor" shape dims x))))))
          (cond ((member -1 dims)
                 (unless (zerop (mod in-n out-n))
                   (shape-error "cannot reshape ~S (~D elements per ~{~S~^ ~}) to ~S" shape in-n in-f dims))
                 (let ((inferred (dim-product (cons (/ in-n out-n) rest-f))))
                   (values (note (format nil "reshape ~S" dims) (substitute inferred -1 dims))
                           `(mx:reshape ,code (list ,@dims)))))
                ((and (= in-n out-n) (null rest-f))
                 (values (note (format nil "reshape ~S" dims) dims) `(mx:reshape ,code (list ,@dims))))
                (t (shape-error "cannot reshape ~S to ~S: sizes differ" shape dims))))))))

(define-stage transpose (axes shape code)
  (unless (equal (sort (copy-list axes) #'<) (loop for i below (length shape) collect i))
    (shape-error "transpose ~S is not a permutation of the axes of ~S" axes shape))
  (values (note (format nil "transpose ~S" axes) (mapcar (lambda (a) (nth a shape)) axes))
          `(mx:transpose ,code :axes ',axes)))

(macrolet ((reduction (name function)
             `(define-stage ,name (args shape code)
                (destructuring-bind (axis &key keepdims) args
                  (let ((a (normalize-axis axis (length shape))))
                    (values (note (format nil "~(~A~) ~D" ',name axis)
                                  (if keepdims
                                      (substitute-nth a 1 shape)
                                      (append (subseq shape 0 a) (subseq shape (1+ a)))))
                            `(,',function ,code :axis ,axis :keepdims ,keepdims)))))))
  (reduction mean mx:mean)
  (reduction sum mx:sum)
  (reduction max mx:max))

(defun substitute-nth (n value list)
  (loop for x in list for i from 0 collect (if (= i n) value x)))

(define-stage elementwise (args shape code)
  (destructuring-bind (function) args
    (values (note "elementwise" shape) `(funcall ,function ,code))))

(define-stage residual (stages shape code)
  (let ((v (gensym "X")))
    (multiple-value-bind (out body) (apply-stages stages shape v)
      (unless (equal out shape)
        (shape-error "a residual branch must keep the shape ~S, but it gives ~S" shape out))
      (values (note "residual (+)" shape) `(let ((,v ,code)) (mx:add ,v ,body))))))

(define-stage repeat (args shape code)
  (destructuring-bind (n &rest stages) args
    (unless (and (integerp n) (plusp n)) (shape-error "repeat needs a positive integer count, not ~S" n))
    (dotimes (i n)
      (multiple-value-setq (shape code) (apply-stages stages shape code)))
    (values shape code)))

(defparameter *activations*
  '("RELU" "RELU6" "LEAKY-RELU" "ELU" "SELU" "CELU" "GELU" "GELU-APPROX" "GELU-FAST-APPROX"
    "SILU" "MISH" "SOFTPLUS" "SOFTSIGN" "LOG-SIGMOID" "HARDSWISH" "HARD-TANH" "SIGMOID"
    "TANH" "SOFTMAX" "LOG-SOFTMAX"))

(defun apply-stage (stage shape code)
  (let* ((*form* stage)
         (form (if (consp stage) stage (list stage)))
         (head (first form)))
    (unless (symbolp head) (shape-error "a stage is a symbol or a list starting with one"))
    (let ((net (gethash head *nets*))
          (handler (gethash (symbol-name head) *stages*)))
      (cond (net (apply-net head net (rest form) shape code))
            (handler (funcall handler (rest form) shape code))
            ((member (symbol-name head) *activations* :test #'string=)
             (values (note (string-downcase (symbol-name head)) shape)
                     `(,(find-symbol (symbol-name head) :mlx.nn) ,code ,@(rest form))))
            (t (shape-error "unknown stage ~S" head))))))

(defun apply-stages (stages shape code)
  (dolist (s stages (values shape code))
    (multiple-value-setq (shape code) (apply-stage s shape code))))

;;; A DEFNET network as a stage: unify its input spec with SHAPE

(defun substitute-dims (dim bindings)
  (cond ((consp dim) (apply #'dim-op (first dim) (mapcar (lambda (d) (substitute-dims d bindings)) (rest dim))))
        ((symbolp dim) (let ((b (assoc dim bindings))) (if b (cdr b) dim)))
        (t dim)))

(defun apply-net (name net args shape code)
  (destructuring-bind (&key inputs output outputs hyper &allow-other-keys) net
    (unless (= 1 (length inputs))
      (shape-error "~S takes ~D inputs; only one-input networks can be stages" name (length inputs)))
    (when (> (length outputs) 1)
      (shape-error "~S has ~D outputs; only one-output networks can be stages" name (length outputs)))
    (let* ((spec (second (first inputs)))
           (hyper-names (mapcar #'first hyper))
           (bindings (loop for (key value) on args by #'cddr
                           collect (cons (or (find (symbol-name key) hyper-names :key #'symbol-name :test #'string=)
                                             (shape-error "~S has no hyperparameter ~S" name key))
                                         value))))
      (unless (= (length spec) (length shape))
        (shape-error "~S expects rank ~D ~S, got ~S" name (length spec) spec shape))
      (loop for want in spec for have in shape
            do (cond ((integerp want)
                      (unless (eql want have)
                        (shape-error "~S expects ~S, got ~S" name spec shape)))
                     (t (let ((bound (assoc want bindings)))
                          (cond ((null bound)
                                 (when (and (member want hyper-names) (not (static-dim-p have)))
                                   (shape-error "~S's ~S would be ~S, which is only known at run time"
                                                name want have))
                                 (push (cons want have) bindings))
                                ((not (equal (cdr bound) have))
                                 (shape-error "~S expects ~S = ~S, got ~S in ~S" name want (cdr bound) have shape)))))))
      ;; unbound hyperparameters take their defaults
      (loop for (h default) in hyper
            unless (assoc h bindings) do (push (cons h default) bindings))
      (let ((child (add-layer name `(,name ,@(loop for h in hyper-names
                                                   collect (intern (symbol-name h) :keyword)
                                                   collect (cdr (assoc h bindings)))))))
        (values (note (format nil "~(~A~)" name) (mapcar (lambda (d) (substitute-dims d bindings)) output) child)
                (call-layer child code))))))

;;; ------------------------------------------------------------------
;;; Expressions

(defun infer (expr env)
  "(values shape code) of EXPR; ENV maps variables to shapes."
  (let ((*form* expr))
    (cond ((numberp expr) (values '() expr))
          ((symbolp expr)
           (let ((b (assoc expr env)))
             (unless b (shape-error "unknown variable ~S" expr))
             (values (cdr b) expr)))
          ((not (and (consp expr) (symbolp (first expr)))) (shape-error "not an expression"))
          ((name= (first expr) "->")
           (multiple-value-bind (shape code) (infer (second expr) env)
             (apply-stages (cddr expr) shape code)))
          ((name= (first expr) "LET*")
           (let ((bindings '()))
             (dolist (b (second expr))
               (destructuring-bind (var value) b
                 (multiple-value-bind (shape code) (infer value env)
                   (push (list var code) bindings)
                   (push (cons var shape) env))))
             (multiple-value-bind (shape code) (infer (third expr) env)
               (values shape `(let* ,(reverse bindings) ,code)))))
          ((member (symbol-name (first expr)) '("+" "-" "*" "/" "MAXIMUM" "MINIMUM") :test #'string=)
           (let ((shape '()) (codes '()))
             (dolist (arg (rest expr))
               (multiple-value-bind (s c) (infer arg env)
                 (setf shape (let ((*form* expr)) (broadcast shape s)))
                 (push c codes)))
             (values (note (format nil "~(~A~)" (first expr)) shape)
                     `(,(cdr (assoc (symbol-name (first expr))
                                    '(("+" . mx:add) ("-" . mx:subtract) ("*" . mx:multiply)
                                      ("/" . mx:divide) ("MAXIMUM" . mx:maximum) ("MINIMUM" . mx:minimum))
                                    :test #'string=))
                       ,@(reverse codes)))))
          ((name= (first expr) "MATMUL")
           (multiple-value-bind (a ac) (infer (second expr) env)
             (multiple-value-bind (b bc) (infer (third expr) env)
               (when (or (< (length a) 2) (< (length b) 2))
                 (shape-error "matmul needs rank 2 or more: ~S and ~S" a b))
               (unless (equal (car (last a)) (car (last b 2)))
                 (shape-error "matmul inner dimensions differ: ~S x ~S" a b))
               (values (note "matmul" (append (broadcast (butlast a 2) (butlast b 2))
                                              (list (car (last a 2)) (car (last b)))))
                       `(mx:matmul ,ac ,bc)))))
          ((name= (first expr) "CONCAT")
           (let* ((axis (second expr))
                  (parts (loop for e in (cddr expr) collect (multiple-value-list (infer e env))))
                  (shapes (mapcar #'first parts))
                  (rank (length (first shapes)))
                  (a (normalize-axis axis rank)))
             (dolist (s (rest shapes))
               (unless (and (= (length s) rank)
                            (loop for x in s for y in (first shapes) for i from 0
                                  always (or (= i a) (equal x y))))
                 (shape-error "concat along ~D needs matching shapes: ~S vs ~S" axis (first shapes) s)))
             (values (note (format nil "concat ~D" axis)
                           (substitute-nth a (apply #'dim-op '+ (mapcar (lambda (s) (nth a s)) shapes))
                                           (first shapes)))
                     `(mx:concatenate (list ,@(mapcar #'second parts)) :axis ,axis))))
          (t (shape-error "~S is not an expression; to apply a layer, write (-> ~S ~S)"
                          (first expr) (second expr) (cons (first expr) (cddr expr)))))))

(defun infer-outputs (expr env)
  "Like INFER for a network's body, which may return several values:
(values e...), possibly as the body of LET*.  Returns (values shapes code)."
  (let ((*form* expr))
    (cond ((and (consp expr) (name= (first expr) "VALUES"))
           (let ((results (loop for e in (rest expr) for i from 1
                                collect (multiple-value-bind (shape code) (infer e env)
                                          (note (format nil "output ~D" i) shape)
                                          (list shape code)))))
             (values (mapcar #'first results) `(values ,@(mapcar #'second results)))))
          ((and (consp expr) (name= (first expr) "LET*"))
           (let ((bindings '()))
             (dolist (b (second expr))
               (destructuring-bind (var value) b
                 (multiple-value-bind (shape code) (infer value env)
                   (push (list var code) bindings)
                   (push (cons var shape) env))))
             (multiple-value-bind (shapes code) (infer-outputs (third expr) env)
               (values shapes `(let* ,(reverse bindings) ,code)))))
          (t (multiple-value-bind (shape code) (infer expr env)
               (values (list shape) code))))))

;;; ------------------------------------------------------------------
;;; Runtime checks

(defun check-input-shapes (net arrays specs hyper-values)
  "Check ARRAYS against SPECS, binding runtime dimensions consistently.
Returns the alist of runtime bindings."
  (let ((bindings (copy-alist hyper-values)))
    (loop for array in arrays for (name spec) in specs
          do (let ((shape (mx:shape array)))
               (unless (= (length shape) (length spec))
                 (error 'nn:shape-error :net net
                                        :message (let ((*print-case* :downcase))
                                                   (format nil "input ~S should have shape ~S, got ~S" name spec shape))))
               (loop for want in spec for have in shape
                     do (let ((bound (if (integerp want) (cons want want) (assoc want bindings))))
                          (cond ((null bound) (push (cons want have) bindings))
                                ((/= (cdr bound) have)
                                 (error 'nn:shape-error
                                        :net net
                                        :message (let ((*print-case* :downcase))
                                                   (format nil "input ~S should have shape ~S, got ~S~
                                                                ~:[ (~S is ~D)~;~]"
                                                           name spec shape (integerp want) want (cdr bound))))))))))
    bindings))

;;; ------------------------------------------------------------------
;;; The macro

(defun runtime-symbols (specs hyper)
  (remove-duplicates
   (loop for (nil spec) in specs
         nconc (loop for d in spec when (and (symbolp d) (not (member d hyper))) collect d))))

(defmacro nn:defnet (name (&rest lambda-list) &body body)
  "Define the network NAME: a module class, a constructor (NAME &key
hyperparameters...) and FORWARD taking the inputs in order.  LAMBDA-LIST
is input specs (var (dim...)), then &key hyperparameters.  BODY is an
optional docstring and one expression.  Shapes are inferred and checked
at macroexpansion time; see src/nn/defnet.lisp for the language."
  (let* ((key-pos (position '&key lambda-list))
         (specs (subseq lambda-list 0 key-pos))
         (hyper (mapcar (lambda (h) (if (consp h) h (list h nil)))
                        (and key-pos (subseq lambda-list (1+ key-pos)))))
         (hyper-names (mapcar #'first hyper))
         (doc (and (stringp (first body)) (rest body) (first body)))
         (expr (car (last body)))
         (*net* name) (*hyper* hyper-names) (*self* (gensym "SELF"))
         (*layers* '()) (*counts* '()) (*trace* '()) (*form* nil)
         (runtime (runtime-symbols specs hyper-names))
         (args (gensym "ARGS")) (bindings (gensym "BINDINGS")))
    (dolist (s specs)
      (unless (and (symbolp (first s)) (listp (second s))
                   (every (lambda (d) (or (integerp d) (symbolp d))) (second s)))
        (error 'nn:shape-error :net name :form s :message "an input is (variable (dimension...))")))
    (dolist (s specs) (note (format nil "input ~(~A~)" (first s)) (second s)))
    (multiple-value-bind (outputs code)
        (infer-outputs expr (mapcar (lambda (s) (cons (first s) (second s))) specs))
      (let ((signature `(:inputs ,specs :output ,(first outputs) :outputs ,outputs
                         :hyper ,hyper :trace ,(reverse *trace*))))
        `(progn
           (eval-when (:compile-toplevel :load-toplevel :execute)
             (setf (gethash ',name *nets*) ',signature))
           (nn:defmodule ,name ()
             ,(loop for (h) in hyper collect `(,h :initarg ,(intern (symbol-name h) :keyword)))
             ,@(when doc `((:documentation ,doc))))
           (defun ,name (&key ,@hyper)
             ,@(when doc (list doc))
             (let ((,*self* (make-instance ',name ,@(loop for h in hyper-names
                                                        collect (intern (symbol-name h) :keyword) collect h))))
               ,@(loop for (child . constructor) in (reverse *layers*)
                       collect `(nn:register ,*self* ,child ,constructor))
               ,*self*))
           (defmethod nn:forward ((,*self* ,name) &rest ,args)
             (destructuring-bind ,(mapcar #'first specs) ,args
               (let* (,@(loop for h in hyper-names collect `(,h (slot-value ,*self* ',h)))
                      (,bindings (check-input-shapes ',name (list ,@(mapcar #'first specs)) ',specs
                                                     (list ,@(loop for h in hyper-names
                                                                   collect `(cons ',h ,h)))))
                      ,@(loop for r in runtime collect `(,r (cdr (assoc ',r ,bindings)))))
                 (declare (ignorable ,bindings ,@hyper-names ,@runtime))
                 ,code)))
           ',name)))))

;;; ------------------------------------------------------------------
;;; Summary

(defun nn:net-summary (net &optional (stream *standard-output*))
  "Print NET's layers (NET is a DEFNET module or name, then made with the
default hyperparameters) with the shape each produces and its parameters."
  (let* ((module (if (symbolp net) (funcall net) net))
         (name (class-name (class-of module)))
         (signature (or (gethash name *nets*) (error "~S was not defined with DEFNET." name)))
         (values (loop for (h) in (getf signature :hyper)
                       collect (cons h (slot-value module h))))
         (*print-case* :downcase)
         (*print-pretty* nil)
         (*package* (or (symbol-package name) *package*)))
    (format stream "~&~A~%~30A ~28A ~12@A~%" name "stage" "shape" "parameters")
    (loop for (description shape child) in (getf signature :trace)
          do (format stream "~30A ~28A ~12@A~%" description
                     (prin1-to-string (mapcar (lambda (d) (substitute-dims d values)) shape))
                     (if child (format nil "~:D" (nn:parameter-count (nn:child module child))) "")))
    (format stream "~30A ~28A ~12:D~%" "total" "" (nn:parameter-count module))
    (values)))
