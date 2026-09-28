;;;; nn/optimizers.lisp -- optimizers, schedules and gradient clipping
;;;; (mlx.optimizers)
;;;;
;;;; Update rules and defaults follow mlx.optimizers exactly (e.g. Adam's
;;;; bias correction is off by default).  An optimizer keeps per-parameter
;;;; state keyed by the parameter's path in the tree, so it works with any
;;;; parameter tree, not just modules.
;;;;
;;;; Typical use:
;;;;   (let ((opt (optim:adam 1e-3))
;;;;         (step (nn:value-and-grad model #'loss)))
;;;;     (dotimes (i n)
;;;;       (mx:with-scope ()
;;;;         (multiple-value-bind (l grads) (funcall step x y)
;;;;           (optim:update opt model grads)
;;;;           (mx:eval (nn:parameters model))))))

(in-package :mlx.nn.impl)

(defclass optim:optimizer ()
  ((learning-rate :initarg :learning-rate
                  :documentation "A number, or a function of the step count (a schedule).")
   (state :initform (make-hash-table :test 'equal) :reader optim:state
          :documentation "Parameter path -> plist of state arrays.")
   (step :initform 0 :reader optim:step-count))
  (:documentation "Base class of optimizers."))

(defgeneric init-single (optimizer parameter)
  (:documentation "Initial state plist for PARAMETER.")
  (:method ((o optim:optimizer) parameter) (declare (ignore parameter)) '()))

(defgeneric apply-single (optimizer gradient parameter state learning-rate)
  (:documentation "Returns (values new-parameter new-state-plist)."))

(defun optim:learning-rate (optimizer)
  "The learning rate for the current step."
  (let ((lr (slot-value optimizer 'learning-rate)))
    (if (functionp lr) (funcall lr (optim:step-count optimizer)) lr)))

(defun (setf optim:learning-rate) (value optimizer)
  "Set a number, or a schedule: a function of the step count."
  (setf (slot-value optimizer 'learning-rate) value))

(defun optim:reset (optimizer)
  "Forget all state and restart the step count."
  (clrhash (slot-value optimizer 'state))
  (setf (slot-value optimizer 'step) 0)
  optimizer)

(defun state-arrays (plist) (loop for (nil v) on plist by #'cddr when (typep v 'mx:mlx-array) collect v))

(defun optim:apply-gradients (optimizer gradients parameters &key (free-old nil))
  "Return a new parameter tree: PARAMETERS updated with GRADIENTS (a tree
of the same shape).  Updates OPTIMIZER's state; with FREE-OLD, frees the
state arrays it replaces."
  (let* ((lr (optim:learning-rate optimizer))
         (state (slot-value optimizer 'state))
         (grads-by-path '()))
    (incf (slot-value optimizer 'step))
    (impl::map-leaves-with-path (lambda (path g) (push (cons path g) grads-by-path)) gradients)
    (setf grads-by-path (nreverse grads-by-path))
    (multiple-value-bind (params structure) (mx:tree-flatten parameters)
      (unless (= (length params) (length grads-by-path))
        (error "Gradient tree (~D leaves) does not match parameters (~D leaves)."
               (length grads-by-path) (length params)))
      (mx:tree-unflatten
       structure
       (loop for (path . g) in grads-by-path
             for p in params
             collect (let ((old (or (gethash path state) (init-single optimizer p))))
                       (multiple-value-bind (new-p new-state) (apply-single optimizer g p old lr)
                         (mx:keep new-p)
                         (apply #'mx:persist (state-arrays new-state))
                         (when free-old
                           (dolist (a (state-arrays old))
                             (unless (member a (state-arrays new-state)) (mx:free a))))
                         (setf (gethash path state) new-state)
                         new-p)))))))

(defun optim:update (optimizer model gradients &key (free-old t))
  "Apply GRADIENTS (shaped like (NN:TRAINABLE-PARAMETERS MODEL)) to MODEL.
The new parameters and state are exempt from the enclosing WITH-SCOPE.
With FREE-OLD (the default) the replaced parameter and state arrays are
freed at once rather than left to the garbage collector -- don't hold on to
them.  Returns MODEL."
  (let* ((params (nn:trainable-parameters model))
         (new (optim:apply-gradients optimizer gradients params :free-old free-old)))
    (nn:update model new)
    (when free-old
      (let ((kept (mx:tree-flatten new)))
        (dolist (a (mx:tree-flatten params))
          (unless (member a kept :test #'eq) (mx:free a)))))
    model))

(defmacro define-optimizer (name (&rest slots) documentation)
  "Define optimizer class NAME with SLOTS ((slot default) ...) and the
constructor (NAME learning-rate &key slot...)."
  `(progn
     (defclass ,name (optim:optimizer)
       ,(mapcar (lambda (s) `(,(first s) :initarg ,(intern (symbol-name (first s)) :keyword)))
                slots))
     (defun ,name (learning-rate &key ,@slots)
       ,documentation
       (make-instance ',name :learning-rate learning-rate
                             ,@(loop for (s) in slots
                                     append (list (intern (symbol-name s) :keyword) s))))))

;;; SGD

(define-optimizer optim:sgd ((momentum 0.0) (weight-decay 0.0) (dampening 0.0) (nesterov nil))
  "Stochastic gradient descent, with optional momentum, Nesterov momentum and weight decay.")

(defmethod init-single ((o optim:sgd) p)
  (when (plusp (slot-value o 'momentum)) (list :v (mx:zeros-like p))))

(defmethod apply-single ((o optim:sgd) g p state lr)
  (with-slots (momentum weight-decay dampening nesterov) o
    (let ((g (if (zerop weight-decay) g (mx:add g (mx:multiply p weight-decay)))))
      (if (<= momentum 0)
          (values (mx:subtract p (mx:multiply g lr)) state)
          (let* ((v (mx:multiply (getf state :v) momentum))
                 (v (if (plusp dampening)
                        (mx:add v (mx:multiply g (- 1 dampening)))
                        (mx:add v g)))
                 (update (if nesterov (mx:add g (mx:multiply v momentum)) v)))
            (values (mx:subtract p (mx:multiply update lr)) (list :v v)))))))

;;; RMSprop, Adagrad, AdaDelta

(define-optimizer optim:rmsprop ((alpha 0.99) (eps 1e-8))
  "RMSprop.")

(defmethod init-single ((o optim:rmsprop) p) (list :v (mx:zeros-like p)))

(defmethod apply-single ((o optim:rmsprop) g p state lr)
  (with-slots (alpha eps) o
    (let ((v (mx:add (mx:multiply (getf state :v) alpha) (mx:multiply (mx:square g) (- 1 alpha)))))
      (values (mx:subtract p (mx:divide (mx:multiply g lr) (mx:add (mx:sqrt v) eps)))
              (list :v v)))))

(define-optimizer optim:adagrad ((eps 1e-8))
  "Adagrad.")

(defmethod init-single ((o optim:adagrad) p) (list :v (mx:zeros-like p)))

(defmethod apply-single ((o optim:adagrad) g p state lr)
  (let ((v (mx:add (getf state :v) (mx:square g))))
    (values (mx:subtract p (mx:divide (mx:multiply g lr) (mx:add (mx:sqrt v) (slot-value o 'eps))))
            (list :v v))))

(define-optimizer optim:adadelta ((rho 0.9) (eps 1e-6))
  "AdaDelta.")

(defmethod init-single ((o optim:adadelta) p) (list :v (mx:zeros-like p) :u (mx:zeros-like p)))

(defmethod apply-single ((o optim:adadelta) g p state lr)
  (with-slots (rho eps) o
    (let* ((v (mx:add (mx:multiply (getf state :v) rho) (mx:multiply (mx:square g) (- 1 rho))))
           (d (mx:multiply (mx:divide (mx:sqrt (mx:add (getf state :u) eps)) (mx:sqrt (mx:add v eps))) g))
           (u (mx:add (mx:multiply (getf state :u) rho) (mx:multiply (mx:square d) (- 1 rho)))))
      (values (mx:subtract p (mx:multiply d lr)) (list :v v :u u)))))

;;; Adam family

(define-optimizer optim:adam ((betas '(0.9 0.999)) (eps 1e-8) (bias-correction nil))
  "Adam.  As in MLX, bias correction is off unless BIAS-CORRECTION.")

(defmethod init-single ((o optim:adam) p) (list :m (mx:zeros-like p) :v (mx:zeros-like p)))

(defun adam-step (o g p state lr)
  (with-slots (betas eps bias-correction) o
    (destructuring-bind (b1 b2) betas
      (let* ((m (mx:add (mx:multiply (getf state :m) b1) (mx:multiply g (- 1 b1))))
             (v (mx:add (mx:multiply (getf state :v) b2) (mx:multiply (mx:square g) (- 1 b2))))
             (new (if bias-correction
                      (let* ((step (optim:step-count o))
                             (c1 (/ lr (- 1 (expt b1 step))))
                             (c2 (/ 1 (sqrt (- 1 (expt b2 step))))))
                        (mx:subtract p (mx:divide (mx:multiply m c1)
                                                  (mx:add (mx:multiply (mx:sqrt v) c2) eps))))
                      (mx:subtract p (mx:divide (mx:multiply m lr) (mx:add (mx:sqrt v) eps))))))
        (values new (list :m m :v v))))))

(defmethod apply-single ((o optim:adam) g p state lr) (adam-step o g p state lr))

(defclass optim:adamw (optim:adam) ((weight-decay :initarg :weight-decay)))

(defun optim:adamw (learning-rate &key (betas '(0.9 0.999)) (eps 1e-8) (weight-decay 0.01)
                                       (bias-correction nil))
  "Adam with decoupled weight decay."
  (make-instance 'optim:adamw :learning-rate learning-rate :betas betas :eps eps
                              :weight-decay weight-decay :bias-correction bias-correction))

(defmethod apply-single ((o optim:adamw) g p state lr)
  (adam-step o g (mx:multiply p (- 1 (* lr (slot-value o 'weight-decay)))) state lr))

(define-optimizer optim:adamax ((betas '(0.9 0.999)) (eps 1e-8))
  "Adamax, Adam with the infinity norm.")

(defmethod init-single ((o optim:adamax) p) (list :m (mx:zeros-like p) :v (mx:zeros-like p)))

(defmethod apply-single ((o optim:adamax) g p state lr)
  (with-slots (betas eps) o
    (destructuring-bind (b1 b2) betas
      (let ((m (mx:add (mx:multiply (getf state :m) b1) (mx:multiply g (- 1 b1))))
            (v (mx:maximum (mx:multiply (getf state :v) b2) (mx:abs g))))
        (values (mx:subtract p (mx:divide (mx:multiply m lr) (mx:add v eps)))
                (list :m m :v v))))))

(define-optimizer optim:lion ((betas '(0.9 0.99)) (weight-decay 0.0))
  "Lion.  Use a learning rate 3-10x smaller than for AdamW.")

(defmethod init-single ((o optim:lion) p) (list :m (mx:zeros-like p)))

(defmethod apply-single ((o optim:lion) g p state lr)
  (with-slots (betas weight-decay) o
    (destructuring-bind (b1 b2) betas
      (let* ((m (getf state :m))
             (c (mx:add (mx:multiply m b1) (mx:multiply g (- 1 b1))))
             (m (mx:add (mx:multiply m b2) (mx:multiply g (- 1 b2))))
             (p (if (plusp weight-decay) (mx:multiply p (- 1 (* lr weight-decay))) p)))
        (values (mx:subtract p (mx:multiply (mx:sign c) lr)) (list :m m))))))

;;; ------------------------------------------------------------------
;;; Schedules: functions from the step count to a learning rate

(defun optim:exponential-decay (initial decay-rate)
  (lambda (step) (float (* initial (expt decay-rate step)) 1.0)))

(defun optim:step-decay (initial decay-rate step-size)
  (lambda (step) (float (* initial (expt decay-rate (floor step step-size))) 1.0)))

(defun optim:cosine-decay (initial decay-steps &key (end 0.0))
  (lambda (step)
    (let ((s (min step decay-steps)))
      (float (+ end (* 0.5 (- initial end) (+ 1 (cos (/ (* pi s) decay-steps))))) 1.0))))

(defun optim:linear-schedule (initial end steps)
  (lambda (step)
    (float (if (< step steps)
               (+ initial (* (- end initial) (/ step steps)))
               end)
           1.0)))

(defun optim:join-schedules (schedules boundaries)
  "Use (first SCHEDULES) until (first BOUNDARIES), then the next schedule
(counting steps from that boundary), and so on."
  (unless (= (length boundaries) (1- (length schedules)))
    (error "Need one boundary fewer than schedules."))
  (lambda (step)
    (loop for schedule in schedules
          for start = 0 then boundary
          for boundary in (append boundaries (list nil))
          when (or (null boundary) (< step boundary))
            return (funcall schedule (- step start)))))

;;; ------------------------------------------------------------------

(defun optim:clip-grad-norm (gradients max-norm)
  "Scale GRADIENTS (a tree) so their global L2 norm is at most MAX-NORM.
Returns (values clipped-gradients total-norm)."
  (let* ((leaves (mx:tree-flatten gradients))
         (total (mx:sqrt (reduce #'mx:add (mapcar (lambda (g) (mx:sum (mx:square g))) leaves))))
         (scale (mx:minimum (mx:divide max-norm (mx:add total 1e-6)) 1.0)))
    (values (mx:tree-map (lambda (g) (mx:multiply g scale)) gradients)
            total)))
