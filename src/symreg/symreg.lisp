;;;; symreg.lisp -- symbolic regression: evolving Lisp expressions on the GPU
;;;;
;;;; Candidate formulas are s-expressions such as (+ (* 2.5 (sin x0)) x1).
;;;; Genetic programming breeds them on the Lisp side; the GPU does the
;;;; arithmetic.  Each generation the whole population is compiled to fixed
;;;; size postfix programs and run by one vectorized stack machine: a single
;;;; MLX graph, compiled once, that evaluates every program on every sample
;;;; at once.  The same graph is differentiated with respect to the programs'
;;;; constants, so Adam tunes the constants of all candidates together
;;;; before they are judged.

(in-package :mlx.symreg)

;;; ------------------------------------------------------------------
;;; Operators
;;;
;;; Every operator is total: division by (nearly) zero gives 1, LOG and
;;; SQRT take the absolute value, EXP saturates, and every result is
;;; clamped to +/-1e7.  So no candidate produces NaN or infinity, on the
;;; GPU (where a NaN in one branch of WHERE would poison the gradient) or
;;; in Lisp, and both compute the same function.

(defparameter +bound+ 1f7)

(declaim (inline bound))
(defun bound (v) (max (- +bound+) (min +bound+ v)))

(defun square (x) (* x x))
(defun protected-div (a b) (if (< (abs b) 1f-6) 1f0 (/ a b)))
(defun protected-log (x) (log (max (abs x) 1f-6)))
(defun protected-sqrt (x) (sqrt (max (abs x) 1f-12)))
(defun protected-exp (x) (exp (min x 50f0)))

(defun mlx-div (a b)
  (let ((small (mx:less (mx:abs b) 1f-6)))
    (mx:where small 1f0 (mx:divide a (mx:where small 1f0 b)))))
(defun mlx-log (x) (mx:log (mx:maximum (mx:abs x) 1f-6)))
(defun mlx-sqrt (x) (mx:sqrt (mx:maximum (mx:abs x) 1f-12)))
(defun mlx-exp (x) (mx:exp (mx:minimum x 50f0)))

(defstruct (operator (:constructor make-operator (name arity lisp mlx)))
  name arity lisp mlx)

(defparameter *operators*
  (list (make-operator '+ 2 '+ #'mx:add)
        (make-operator '- 2 '- #'mx:subtract)
        (make-operator '* 2 '* #'mx:multiply)
        (make-operator '/ 2 'protected-div #'mlx-div)
        (make-operator 'sin 1 'sin #'mx:sin)
        (make-operator 'cos 1 'cos #'mx:cos)
        (make-operator 'exp 1 'protected-exp #'mlx-exp)
        (make-operator 'log 1 'protected-log #'mlx-log)
        (make-operator 'sqrt 1 'protected-sqrt #'mlx-sqrt)
        (make-operator 'square 1 'square #'mx:square)
        (make-operator 'tanh 1 'tanh #'mx:tanh))
  "Every operator available.  LISP names the Lisp function computing it.")

(defparameter *default-operators* '(+ - * / sin cos exp log sqrt square))

(defun operator-names () (mapcar #'operator-name *operators*))

(defun find-operator (name)
  (or (find name *operators* :key #'operator-name)
      (find (string name) *operators* :key (lambda (o) (string (operator-name o))) :test #'string-equal)
      (error "Unknown operator ~S; known: ~{~A~^ ~}." name (operator-names))))

;;; ------------------------------------------------------------------
;;; Expressions: numbers (constants), symbols (variables), (op args...)

(defun expression-size (e)
  (if (atom e) 1 (1+ (reduce #'+ (rest e) :key #'expression-size))))

(defun stack-need (e)
  "Stack slots needed to evaluate E in postfix order."
  (cond ((atom e) 1)
        ((null (cddr e)) (stack-need (second e)))
        (t (max (stack-need (second e)) (1+ (stack-need (third e)))))))

(defun lisp-code (e)
  (cond ((numberp e) (float e 1f0))
        ((symbolp e) e)
        (t `(bound (,(operator-lisp (find-operator (first e))) ,@(mapcar #'lisp-code (rest e)))))))

(defun expression-function (expression variables)
  "A compiled Lisp function of VARIABLES computing EXPRESSION (in single
floats, with the protected operators)."
  (let ((fn (compile nil `(lambda ,variables
                            (let ,(loop for v in variables collect `(,v (float ,v 1f0)))
                              (declare (ignorable ,@variables))
                              ,(lisp-code expression))))))
    (lambda (&rest args)
      (sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero :inexact :underflow)
        (apply fn args)))))

(defun expression->mlx (expression variables inputs)
  "EXPRESSION as an MLX graph: VARIABLES are bound to the arrays INPUTS."
  (labels ((walk (e)
             (cond ((numberp e) (mx:from-lisp (float e 1f0)))
                   ((symbolp e) (or (nth (position e variables) inputs)
                                    (error "Unbound variable ~S." e)))
                   (t (mx:clip (apply (operator-mlx (find-operator (first e))) (mapcar #'walk (rest e)))
                               :a-min (- +bound+) :a-max +bound+)))))
    (walk expression)))

(defun simplify-expression (e)
  "E with constant subexpressions folded."
  (if (atom e)
      e
      (let ((args (mapcar #'simplify-expression (rest e))))
        (if (every #'numberp args)
            (sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero :inexact :underflow)
              (float (bound (apply (operator-lisp (find-operator (first e))) args)) 1f0))
            (cons (first e) args)))))

(defun round-constants (e &optional (digits 5))
  (cond ((floatp e)
         (if (zerop e)
             0.0
             (let ((scale (expt 10 (- digits 1 (floor (log (abs e) 10))))))
               (/ (fround (* e scale)) scale))))
        ((atom e) e)
        (t (cons (first e) (mapcar (lambda (x) (round-constants x digits)) (rest e))))))

;;; ------------------------------------------------------------------
;;; Physical units
;;;
;;; A unit is an alist of (base . exponent), sorted, e.g. ((:KG . 1)
;;; (:M . 1) (:S . -2)); NIL is dimensionless.  Written as a string
;;; "kg*m/s^2" or a list (kg m (s -2)).  Units rule out formulas that add
;;; metres to kilograms or take the sine of a mass.  As in PySR, a
;;; constant may carry any unit (:ANY), so (* 2.0 mass) can be added to
;;; a length: the constant is a length per mass.

(defun normalize-unit (alist)
  (let ((table '()))
    (loop for (base . exponent) in alist
          do (let ((cell (assoc base table)))
               (if cell (incf (cdr cell) exponent) (push (cons base exponent) table))))
    (sort (remove-if #'zerop table :key #'cdr) #'string< :key (lambda (c) (symbol-name (car c))))))

(defun parse-unit (unit)
  "UNIT (a string such as \"kg*m/s^2\" or \"m^(1/2)\", \"1\" or \"\", or a
list such as (kg m (s -2))) as a normalized alist."
  (flet ((base (name) (intern (string-upcase (string-trim " " (string name))) :keyword)))
    (etypecase unit
      (null nil)
      (cons (normalize-unit (mapcar (lambda (u) (if (consp u) (cons (base (first u)) (rational (second u)))
                                                   (cons (base u) 1)))
                                    unit)))
      (string
       (let* ((slash (let ((depth 0))    ; the division, not a slash in "m^(1/2)"
                       (position-if (lambda (c) (case c
                                                  (#\( (incf depth) nil)
                                                  (#\) (decf depth) nil)
                                                  (#\/ (zerop depth))))
                                    unit)))
              (parts (list (cons (subseq unit 0 (or slash (length unit))) 1)
                           (and slash (cons (subseq unit (1+ slash)) -1)))))
         (normalize-unit
          (loop for (text . sign) in (remove nil parts)
                nconc (loop for factor in (uiop:split-string text :separator "*")
                            for trimmed = (string-trim " " factor)
                            for caret = (position #\^ trimmed)
                            for name = (subseq trimmed 0 (or caret (length trimmed)))
                            unless (or (string= name "") (string= name "1"))
                              collect (cons (base name)
                                            (* sign (if caret
                                                        (let ((*read-eval* nil))
                                                          (rational (read-from-string
                                                                     (string-trim "()" (subseq trimmed (1+ caret))))))
                                                        1))))))))
      (symbol (parse-unit (list unit))))))

(defun unit-string (unit)
  (if (null unit)
      "1"
      (format nil "~{~A~^*~}"
              (loop for (base . exponent) in unit
                    collect (if (= exponent 1)
                                (string-downcase base)
                                (format nil "~(~A~)^~A" base exponent))))))

(defun expression-units (e units)
  "The unit of expression E given UNITS, an alist from variable to unit:
a unit, :ANY (it has a free constant factor) or :INVALID."
  (labels ((unit (e)
             (cond ((numberp e) :any)
                   ((symbolp e) (let ((cell (assoc e units))) (if cell (cdr cell) nil)))
                   (t (let ((args (mapcar #'unit (rest e))))
                        (if (member :invalid args)
                            :invalid
                            (combine (first e) args))))))
           (scale (u k) (if (eq u :any) :any (normalize-unit (mapcar (lambda (c) (cons (car c) (* k (cdr c)))) u))))
           (combine (op args)
             (destructuring-bind (a &optional b) args
               (case op
                 ((+ -) (cond ((eq a :any) b) ((eq b :any) a) ((equal a b) a) (t :invalid)))
                 (* (if (or (eq a :any) (eq b :any)) :any (normalize-unit (append a b))))
                 (/ (if (or (eq a :any) (eq b :any)) :any (normalize-unit (append a (scale b -1)))))
                 (square (scale a 2))
                 (sqrt (scale a 1/2))
                 ;; transcendental functions need a dimensionless argument
                 (t (if (or (eq a :any) (null a)) nil :invalid))))))
    (unit e)))

;;; ------------------------------------------------------------------
;;; The stack machine
;;;
;;; A program is a row of instructions: 0 no-op, 1 push constant, 2..V+1
;;; push variable, then one code per operator.  For each instruction we
;;; also record the stack depth before it (operands are read from slots
;;; depth-2) and the slot its result goes to (for a no-op, the last slot,
;;; unused by then).  The last operand is the previous instruction's result.
;;; The stack is a (P S N) array: P programs, S slots, N samples.

(defstruct (machine (:constructor %make-machine))
  variables operators length stack-size (scaling t))

(defun encode-population (machine expressions)
  "Lisp arrays (ops consts depths slots), each (P L), for EXPRESSIONS."
  (let* ((p (length expressions))
         (l (machine-length machine))
         (vars (machine-variables machine))
         (ops (machine-operators machine))
         (codes (make-array (list p l) :element-type '(signed-byte 32) :initial-element 0))
         (consts (make-array (list p l) :element-type 'single-float :initial-element 0f0))
         (depths (make-array (list p l) :element-type '(signed-byte 32) :initial-element 0))
         (slots (make-array (list p l) :element-type '(signed-byte 32) :initial-element -1)))
    (loop for e in expressions for row from 0
          do (let ((i 0) (depth 0))
               (labels ((emit (code const new-depth slot)
                          (setf (aref codes row i) code (aref consts row i) const
                                (aref depths row i) depth (aref slots row i) slot
                                depth new-depth)
                          (incf i))
                        (walk (e)
                          (cond ((numberp e) (emit 1 (float e 1f0) (1+ depth) depth))
                                ((symbolp e) (emit (+ 2 (or (position e vars) (error "Unknown variable ~S." e)))
                                                   0f0 (1+ depth) depth))
                                (t (mapc #'walk (rest e))
                                   (let ((code (+ 2 (length vars) (position (first e) ops :key #'operator-name))))
                                     (if (null (cddr e))
                                         (emit code 0f0 depth (1- depth))
                                         (emit code 0f0 (1- depth) (- depth 2))))))))
                 (walk e)
                 (loop for j from i below l
                       do (setf (aref depths row j) depth
                                (aref slots row j) (1- (machine-stack-size machine)))))))
    (values codes consts depths slots)))

(defun run-machine (machine codes consts depths slots x)
  "Predictions (P N) of the encoded programs on X (V N), as MLX arrays.
An operator's last operand is always the result of the instruction just
before it (the program is postfix), so only a first operand is read back
from the stack, with a gather."
  (let* ((p (mx:dim codes 0))
         (n (mx:dim x 1))
         (s (machine-stack-size machine))
         (nvars (length (machine-variables machine)))
         (slot-ids (mx:reshape (mx:arange s) (list 1 s 1)))
         (var-codes (mx:reshape (mx:arange 2 (+ 2 nvars)) (list 1 1 nvars)))
         ;; every push-variable instruction's value, (P L N), in one matmul
         (leaves (mx:matmul (mx:astype (mx:equal (mx:expand-dims codes 2) var-codes) :float32) x))
         (stack (mx:zeros (list p s n)))
         (previous (mx:zeros (list p n))))
    (dotimes (i (machine-length machine))
      (let* ((column (lambda (a) (mx:reshape (mx:ref a t i) (list p 1 1))))  ; (P 1 1)
             (code (mx:reshape (funcall column codes) (list p 1)))
             (below (mx:maximum (mx:subtract (funcall column depths) 2) 0))
             (next (mx:squeeze-axis
                    (mx:take-along-axis stack (mx:broadcast-to below (list p 1 n)) 1) 1))
             (top previous)
             (value (mx:where (mx:equal code 1)
                              (mx:reshape (funcall column consts) (list p 1))
                              (mx:ref leaves t i t))))
        (loop for op in (machine-operators machine)
              for k from (+ 2 nvars)
              do (setf value (mx:where (mx:equal code k)
                                       (if (= (operator-arity op) 1)
                                           (funcall (operator-mlx op) top)
                                           (funcall (operator-mlx op) next top))
                                       value)))
        (setf value (mx:clip value :a-min (- +bound+) :a-max +bound+)
              stack (mx:where (mx:equal slot-ids (funcall column slots)) (mx:expand-dims value 1) stack)
              previous value)))
    (mx:ref stack t 0 t)))

(defun machine-fit (machine codes consts depths slots x y)
  "For each program f, (list mse a b): its mean squared error as a + b f,
never NaN.  With linear scaling (Keijzer), a and b are the least-squares
offset and scale, computed in closed form; otherwise 0 and 1.  Scaling
spares evolution from having to find a formula's outer constants."
  (let ((f (run-machine machine codes consts depths slots x)))
    (flet ((mse (prediction)
             (mx:nan-to-num (mx:mean (mx:square (mx:subtract prediction y)) :axis 1)
                            :nan 1f30 :posinf 1f30)))
      (if (not (machine-scaling machine))
          (list (mse f) (mx:zeros (list (mx:dim f 0))) (mx:ones (list (mx:dim f 0))))
          (let* ((f-mean (mx:mean f :axis 1 :keepdims t))
                 (y-mean (mx:mean y :axis 1 :keepdims t))  ; per row: rows may fit different targets
                 (df (mx:subtract f f-mean))
                 (var (mx:mean (mx:square df) :axis 1 :keepdims t))
                 (flat (mx:less var 1f-12))
                 (b (mx:where flat 0f0 (mx:divide (mx:mean (mx:multiply df (mx:subtract y y-mean))
                                                           :axis 1 :keepdims t)
                                                  (mx:where flat 1f0 var))))
                 (a (mx:subtract y-mean (mx:multiply b f-mean))))
            (list (mse (mx:add a (mx:multiply b f)))
                  (mx:reshape a (list -1))
                  (mx:reshape b (list -1))))))))

(defun machine-losses (machine codes consts depths slots x y)
  (first (machine-fit machine codes consts depths slots x y)))

(defun make-tuner (machine learning-rate)
  "A compiled Adam step over every program's constants at once.  Tracks
the best constants seen per program, since a step can make things worse."
  (let ((b1 0.9f0) (b2 0.999f0))
    (mx:compile
     (lambda (consts m v best best-loss step codes depths slots x y)
       (multiple-value-bind (out grad)
           (funcall (mx:value-and-grad
                     (lambda (c)
                       (let ((losses (machine-losses machine codes c depths slots x y)))
                         (list (mx:sum losses) losses))))
                    consts)
         (let* ((losses (second out))
                (g (mx:nan-to-num grad :nan 0f0 :posinf 0f0 :neginf 0f0))
                (better (mx:less losses best-loss))
                (m (mx:add (mx:multiply b1 m) (mx:multiply (- 1 b1) g)))
                (v (mx:add (mx:multiply b2 v) (mx:multiply (- 1 b2) (mx:square g))))
                (m-hat (mx:divide m (mx:subtract 1f0 (mx:power b1 step))))
                (v-hat (mx:divide v (mx:subtract 1f0 (mx:power b2 step)))))
           (list (mx:subtract consts (mx:divide (mx:multiply learning-rate m-hat)
                                                (mx:add (mx:sqrt v-hat) 1f-8)))
                 m v
                 (mx:where (mx:expand-dims better 1) consts best)
                 (mx:where better losses best-loss))))))))

(defun write-back-constants (expression row consts)
  "EXPRESSION with its constants replaced, in postfix order, by the
entries of CONSTS row ROW at the positions of push-constant instructions."
  (let ((i 0))
    (labels ((walk (e)
               (cond ((numberp e) (prog1 (aref consts row i) (incf i)))
                     ((symbolp e) (incf i) e)
                     (t (let ((args (mapcar #'walk (rest e))))
                          (incf i)
                          (cons (first e) args))))))
      (walk expression))))

(defun evaluate-expressions (expressions x &key variables (operators *default-operators*))
  "Predictions of every expression on X (a Lisp array (N V) or a sequence
of rows) with the GPU stack machine: a Lisp array (P N)."
  (multiple-value-bind (xs n v) (data-matrix x)
    (declare (ignore n))
    (let* ((variables (or variables (default-variables v)))
           (machine (%make-machine :variables variables
                                   :operators (mapcar #'find-operator operators)
                                   :length (reduce #'max expressions :key #'expression-size)
                                   :stack-size (reduce #'max expressions :key #'stack-need))))
      (mx:with-scope ()
        (multiple-value-bind (codes consts depths slots) (encode-population machine expressions)
          (mx:to-lisp (run-machine machine (mx:from-lisp codes) (mx:from-lisp consts)
                                   (mx:from-lisp depths) (mx:from-lisp slots) (mx:from-lisp xs))))))))

;;; ------------------------------------------------------------------
;;; Data

(defun default-variables (n)
  (loop for i below n collect (intern (format nil "X~D" i) :mlx.symreg)))

(defun data-matrix (x)
  "X (a 2D array of samples by variables, a sequence of rows, or a vector
of single values) as a (V N) single-float array; also returns N and V."
  (let* ((rows (cond ((and (arrayp x) (= (array-rank x) 2))
                      (loop for i below (array-dimension x 0)
                            collect (loop for j below (array-dimension x 1) collect (aref x i j))))
                     (t (map 'list (lambda (r) (if (numberp r) (list r) (coerce r 'list))) x))))
         (n (length rows))
         (v (length (first rows)))
         (m (make-array (list v n) :element-type 'single-float)))
    (loop for row in rows for i from 0
          do (loop for value in row for j from 0 do (setf (aref m j i) (float value 1f0))))
    (values m n v)))

;;; ------------------------------------------------------------------
;;; Genetic programming

(defun random-elt (list) (nth (random (length list)) list))

(defun random-constant () (- (random 4f0) 2f0))

(defun random-leaf (variables)
  (if (< (random 1f0) 0.6) (random-elt variables) (random-constant)))

(defun random-expression (depth full variables operators)
  (if (or (zerop depth) (and (not full) (< (random 1f0) 0.3)))
      (random-leaf variables)
      (let ((op (random-elt operators)))
        (cons (operator-name op)
              (loop repeat (operator-arity op)
                    collect (random-expression (1- depth) full variables operators))))))

(defun subexpression (e index)
  "The INDEXth subexpression of E, in preorder."
  (let ((i index))
    (labels ((walk (e)
               (when (zerop i) (return-from subexpression e))
               (decf i)
               (unless (atom e) (mapc #'walk (rest e)))))
      (walk e))))

(defun replace-subexpression (e index new)
  (let ((i index))
    (labels ((walk (e)
               (cond ((zerop (prog1 i (decf i))) new)
                     ((atom e) e)
                     (t (cons (first e) (mapcar #'walk (rest e)))))))
      (walk e))))

(defun mutate (e variables operators)
  (let* ((index (random (expression-size e)))
         (old (subexpression e index)))
    (case (random-elt '(:subtree :subtree :point :point :constant :hoist))
      (:subtree (replace-subexpression e index (random-expression (1+ (random 3)) nil variables operators)))
      (:point (replace-subexpression
               e index
               (if (atom old)
                   (random-leaf variables)
                   (let ((same (remove (length (rest old)) operators :key #'operator-arity :test #'/=)))
                     (cons (operator-name (random-elt same)) (rest old))))))
      (:constant (let ((positions (loop for i below (expression-size e)
                                        when (numberp (subexpression e i)) collect i)))
                   (if positions
                       (let ((i (random-elt positions)))
                         (replace-subexpression e i (* (subexpression e i) (+ 0.5 (random 1f0)))))
                       (replace-subexpression e index (random-constant)))))
      (:hoist (if (atom old) e old)))))

(defun crossover (a b)
  (replace-subexpression a (random (expression-size a))
                         (subexpression b (random (expression-size b)))))

(defstruct (candidate (:constructor make-candidate (expression loss size &optional (genome expression))))
  expression loss size
  genome)                               ; the evolved expression, before scaling

(defun scaled-expression (genome a b &optional (variance 1f0))
  "GENOME as a + b GENOME, leaving out a unit scale, and an offset too
small to matter: one adding at most 1e-9 to the loss (MSE / VARIANCE)."
  (let ((term (cond ((< (abs b) 1f-12) nil)
                    ((< (abs (- b 1)) 1f-6) genome)
                    (t (list '* b genome)))))
    (cond ((null term) a)
          ((<= (* a a) (* 1f-9 variance)) term)
          (t (list '+ a term)))))

(defmethod print-object ((c candidate) stream)
  (print-unreadable-object (c stream :type t)
    (format-expression-line stream "size ~D loss ~,3,,,,,'EG ~S" (candidate-size c) (candidate-loss c)
            (round-constants (candidate-expression c)))))

(defun pareto-front (hall target-loss)
  "The candidates in HALL (a hash table size -> candidate) that beat every
smaller one, by size.  Beating means a loss at least 5% lower, and none
beats a loss at or below TARGET-LOSS: rounding noise is not progress."
  (let ((best most-positive-single-float))
    (loop for c in (sort (loop for c being the hash-values of hall collect c) #'< :key #'candidate-size)
          when (and (> best target-loss) (< (candidate-loss c) (* 0.95 best)))
            collect (progn (setf best (candidate-loss c)) c))))

(defun target-columns (y n)
  "Y as a list of target vectors of single floats: Y is a sequence of N
numbers (one target), or of N rows, or an N x K array (K targets)."
  (let ((rows (if (and (arrayp y) (= (array-rank y) 2))
                  (loop for i below (array-dimension y 0)
                        collect (loop for j below (array-dimension y 1) collect (aref y i j)))
                  (map 'list #'identity y))))
    (unless (= n (length rows))
      (error "X has ~D samples but Y has ~D." n (length rows)))
    (if (numberp (first rows))
        (list (map 'vector (lambda (v) (float v 1f0)) rows))
        (loop for k below (length (first rows))
              collect (map 'vector (lambda (row) (float (elt row k) 1f0)) rows)))))

(defun symbolic-regression (x y &key variables (operators *default-operators*)
                                     units target-units
                                     (population 1000) (generations 100)
                                     (max-size 30) (stack-size 10)
                                     (tuning-steps 8) (learning-rate 0.1)
                                     (parsimony 1e-3) (tournament 5) (elites 0.02)
                                     (scaling t) (max-samples 1024)
                                     (target-loss 1e-9) (patience 10) time-limit
                                     seed (stream *standard-output*))
  "Search for a formula in VARIABLES (default X0, X1, ...) fitting Y given
X (samples by variables: a 2D array, a sequence of rows, or a vector for
one variable).  Y is a sequence of targets, or of rows (or an N x K array)
for K targets, which are fitted at once in one GPU batch.

Each generation breeds POPULATION expressions (per target) of at most
MAX-SIZE nodes from OPERATORS (see OPERATOR-NAMES), tunes all their
constants on the GPU with TUNING-STEPS of Adam, and selects by tournament
on normalized mean squared error (MSE / variance of the target) plus
PARSIMONY per node.  At most MAX-SAMPLES samples (a random subset) are
used.  With SCALING, each candidate f is judged as a + b f with the best
offset and scale, found in closed form.  Stops after GENERATIONS or
TIME-LIMIT seconds, or PATIENCE generations after every target's loss
first reaches TARGET-LOSS.

UNITS gives the variables' physical units, a list parallel to VARIABLES
(each as PARSE-UNIT reads it, NIL for dimensionless): only dimensionally
consistent formulas are bred.  Without SCALING, the formula's unit must
also be TARGET-UNITS (one unit, or a list with one per target).

Returns (values best front): FRONT lists, by size, the candidates that fit
better than every smaller one (a Pareto front of size against loss); BEST
is the one with the lowest loss plus PARSIMONY per node.  With several
targets, both values are lists, one element per target."
  (let* ((*random-state* (if seed (sb-ext:seed-random-state seed) (make-random-state t)))
         (operators (mapcar #'find-operator operators)))
    (multiple-value-bind (xs n v) (data-matrix x)
      (let ((targets (target-columns y n)))
        (when (> n max-samples)           ; a fixed random subset
          (let* ((keep (subseq (shuffle (loop for i below n collect i)) 0 max-samples))
                 (sub (make-array (list v max-samples) :element-type 'single-float)))
            (loop for i in keep for k from 0
                  do (dotimes (j v) (setf (aref sub j k) (aref xs j i))))
            (setf xs sub
                  targets (mapcar (lambda (ys) (map 'vector (lambda (i) (aref ys i)) keep)) targets)
                  n max-samples)))
        (let* ((k (length targets))
               (variables (or variables (default-variables v)))
               (unit-alist (and units (mapcar (lambda (var u) (cons var (parse-unit u))) variables units)))
               (target-unit-list (cond ((null target-units) (make-list k))
                                       ((and (> k 1) (listp target-units) (= (length target-units) k)
                                             (not (and (consp (first target-units)) (numberp (second (first target-units))))))
                                        (mapcar #'parse-unit target-units))
                                       (t (make-list k :initial-element (parse-unit target-units)))))
               (variances (mapcar (lambda (ys)
                                    (let ((mean (/ (reduce #'+ ys) n)))
                                      (max 1f-12 (/ (reduce #'+ ys :key (lambda (a) (expt (- a mean) 2))) n))))
                                  targets))
               (row-variances (coerce (loop for variance in variances
                                            nconc (make-list population :initial-element variance))
                                      'vector))
               (machine (%make-machine :variables variables :operators operators
                                       :length max-size :stack-size stack-size :scaling scaling))
               (tuner (make-tuner machine learning-rate))
               (fit (mx:compile (lambda (codes consts depths slots x y)
                                  (machine-fit machine codes consts depths slots x y))))
               (solved (make-list k :initial-element 0))
               (halls (loop repeat k collect (make-hash-table)))
               (start (get-internal-real-time))
               (x-array (mx:persist (mx:from-lisp xs)))
               ;; every program's own target row: (K*POPULATION N)
               (y-array (mx:persist
                         (mx:with-scope ()
                           (mx:keep (mx:take (mx:from-lisp (mapcar (lambda (ys) (coerce ys 'list)) targets))
                                             (mx:from-lisp (loop for g below k nconc (make-list population :initial-element g))
                                                           :dtype :int32)
                                             :axis 0))))))
          (labels ((valid-p (e target-unit)
                     (and (<= (expression-size e) max-size) (<= (stack-need e) stack-size)
                          (or (null unit-alist)
                              (let ((u (expression-units e unit-alist)))
                                (and (not (eq u :invalid))
                                     (or scaling (eq u :any) (equal u target-unit)))))))
                   (score (c) (+ (candidate-loss c) (* parsimony (candidate-size c))))
                   (best-of (candidates) (reduce (lambda (a b) (if (<= (score a) (score b)) a b)) candidates))
                   (initial (target-unit)
                     (loop for i below population
                           collect (or (loop repeat 1000
                                             for e = (random-expression (+ 1 (mod i 4)) (evenp i) variables operators)
                                             when (valid-p e target-unit) return e)
                                       (or (find-if (lambda (var) (valid-p var target-unit)) variables)
                                           1.0)))))
            (let ((pops (mapcar #'initial target-unit-list)))
              (unwind-protect
                   (loop for generation from 1 to generations
                         do (let* ((all (tune-and-score (reduce #'append pops) machine tuner fit tuning-steps
                                                        x-array y-array row-variances))
                                   (groups (loop for g below k
                                                 collect (subseq all (* g population) (* (1+ g) population))))
                                   (bests (mapcar #'best-of groups)))
                              (loop for candidates in groups for hall in halls
                                    do (dolist (c candidates)
                                         (let ((old (gethash (candidate-size c) hall)))
                                           (when (or (null old) (< (candidate-loss c) (candidate-loss old)))
                                             (setf (gethash (candidate-size c) hall) c)))))
                              (when stream
                                (let ((seconds (/ (- (get-internal-real-time) start) internal-time-units-per-second)))
                                  (if (= k 1)
                                      (format-expression-line stream "~&gen ~3D  loss ~10,3,,,,,'EG  size ~2D  ~,1Fs  ~S~%"
                                                              generation (candidate-loss (first bests))
                                                              (candidate-size (first bests)) seconds
                                                              (round-constants (candidate-expression (first bests)) 4))
                                      (format-expression-line stream "~&gen ~3D  ~,1Fs~{  ~10,3,,,,,'EG~}~%"
                                                              generation seconds (mapcar #'candidate-loss bests))))
                                (force-output stream))
                              ;; once solved, a few more generations let parsimony find a
                              ;; smaller form
                              (setf solved (loop for b in bests for count in solved
                                                 collect (if (<= (candidate-loss b) target-loss) (1+ count) count)))
                              (when (or (every (lambda (count) (> count patience)) solved)
                                        (= generation generations)
                                        (and time-limit
                                             (> (- (get-internal-real-time) start)
                                                (* time-limit internal-time-units-per-second))))
                                (loop-finish))
                              (setf pops (loop for candidates in groups for target-unit in target-unit-list
                                               collect (next-generation candidates population elites tournament
                                                                        #'score (lambda (e) (valid-p e target-unit))
                                                                        variables operators)))))
                (mx:free x-array)
                (mx:free y-array))
              (let* ((fronts (mapcar (lambda (hall) (pareto-front hall target-loss)) halls))
                     (bests (mapcar #'best-of fronts)))
                (if (= k 1)
                    (values (first bests) (first fronts))
                    (values bests fronts))))))))))

(defun format-expression-line (stream control &rest args)
  (let ((*print-pretty* nil) (*print-case* :downcase) (*package* (find-package :mlx.symreg)))
    (apply #'format stream control args)))

(defun shuffle (list)
  (let ((v (coerce list 'vector)))
    (loop for i from (1- (length v)) downto 1
          do (rotatef (aref v i) (aref v (random (1+ i)))))
    (coerce v 'list)))

(defun tune-and-score (expressions machine tuner fit steps x y variances)
  "Tune the constants of all EXPRESSIONS together; candidates with the
tuned constants (and linear scaling) and their losses, normalized by
VARIANCES (a number, or one per expression: the variance of its target)."
  (mx:with-scope ()
    (multiple-value-bind (codes consts depths slots) (encode-population machine expressions)
      (let* ((p (length expressions))
             (codes (mx:from-lisp codes)) (depths (mx:from-lisp depths)) (slots (mx:from-lisp slots))
             (c (mx:from-lisp consts))
             (m (mx:zeros-like c)) (v (mx:zeros-like c))
             (best c)
             (best-loss (mx:full (list p) 1f30)))
        (loop for step from 1 to (max steps 1)
              do (destructuring-bind (c* m* v* best* best-loss*)
                     (funcall tuner c m v best best-loss (float step 1f0) codes depths slots x y)
                   (setf c c* m m* v v* best best* best-loss best-loss*)
                   (mx:eval c m v best best-loss)))
        (destructuring-bind (losses a b) (mapcar #'mx:to-lisp (funcall fit codes best depths slots x y))
          (let ((consts (mx:to-lisp best)))
            (loop for e in expressions for row from 0
                  for variance = (if (numberp variances) variances (elt variances row))
                  collect (let* ((genome (write-back-constants e row consts))
                                 (model (simplify-expression
                                         (scaled-expression genome (aref a row) (aref b row) variance))))
                            (make-candidate model (/ (aref losses row) variance) (expression-size model)
                                            genome)))))))))

(defun next-generation (candidates population elites tournament score valid-p variables operators)
  (let* ((ranked (sort (copy-list candidates) #'< :key score))
         (pool (coerce ranked 'vector))
         (seen (make-hash-table :test 'equal))
         (next '()))
    (flet ((pick ()
             ;; tournament: the best of TOURNAMENT random entrants (POOL is ranked)
             (aref pool (loop repeat tournament minimize (random (length pool)))))
           (add (e) (unless (gethash e seen) (setf (gethash e seen) t) (push e next))))
      (loop for c in ranked repeat (max 1 (round (* elites population)))
            do (add (candidate-genome c)))
      (loop with tries = 0
            while (and (< (length next) population) (< (incf tries) (* 20 population)))
            do (let* ((parent (candidate-genome (pick)))
                      (child (if (< (random 1f0) 0.5)
                                 (crossover parent (candidate-genome (pick)))
                                 (mutate parent variables operators))))
                 (when (funcall valid-p child) (add child))))
      (loop for tries from 0
            while (< (length next) population)
            do (let ((e (if (< tries (* 100 population))
                            (random-expression 3 nil variables operators)
                            (random-constant))))    ; a constant always fits
                 (when (funcall valid-p e) (push e next))))
      (nreverse next))))
