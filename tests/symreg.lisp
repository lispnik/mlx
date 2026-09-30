;;;; tests/symreg.lisp -- mlx/symreg: the GPU stack machine, constant
;;;; tuning, linear scaling and the genetic search

(defpackage :mlx-symreg-tests
  (:use :cl :fiveam)
  (:local-nicknames (:mx :mlx) (:sr :mlx.symreg))
  (:import-from :mlx.symreg #:square)
  (:export #:run-tests))

(in-package :mlx-symreg-tests)

(def-suite :mlx.symreg :description "mlx/symreg tests.")
(in-suite :mlx.symreg)

(defun run-tests ()
  "Run the suite.  $MLX_CL_TEST_DEVICE=cpu runs it on the CPU."
  (let ((device (uiop:getenv "MLX_CL_TEST_DEVICE")))
    (when (and device (plusp (length device)))
      (mx:set-default-device (intern (string-upcase device) :keyword))))
  (let ((results (run :mlx.symreg)))
    (explain! results)
    (results-status results)))

(defparameter *vars* '(x0 x1))

(defun samples (n &key (seed 1) (low -3.0) (high 3.0))
  (let ((*random-state* (sb-ext:seed-random-state seed)))
    (loop repeat n collect (list (+ low (random (- high low))) (+ low (random (- high low)))))))

(defun close-p (a b &optional (tolerance 1e-4))
  (<= (abs (- a b)) (* tolerance (max 1 (abs a) (abs b)))))

(test expression-basics
  (is (= 5 (sr:expression-size '(+ x0 (* 2.0 x1)))))
  (is (= 3 (mlx.symreg::stack-need '(+ x0 (* 2.0 x1)))))
  (is (= 2 (mlx.symreg::stack-need '(+ (* 2.0 x1) x0))))
  (is (= 1 (mlx.symreg::stack-need '(sin (sin x0)))))
  (is (equal '(+ x0 3.0) (sr:simplify-expression '(+ x0 (+ 1.0 2.0)))))
  (is (equal '(* 1.2346 x0) (mlx.symreg::round-constants '(* 1.234567 x0))))
  ;; the protected operators are total
  (let ((f (sr:expression-function '(+ (/ x0 0.0) (log (- x0 x0)) (sqrt (- 0.0 x1))) *vars*)))
    (is (close-p (+ 1 (log 1e-6) 2) (funcall f 5 4)))))

(test genetic-operators-keep-expressions-well-formed
  (let* ((*random-state* (sb-ext:seed-random-state 3))
         (ops (mapcar #'mlx.symreg::find-operator sr:*default-operators*)))
    (labels ((well-formed-p (e)
               (cond ((numberp e) t)
                     ((symbolp e) (member e *vars*))
                     (t (let ((op (mlx.symreg::find-operator (first e))))
                          (and (= (length (rest e)) (mlx.symreg::operator-arity op))
                               (every #'well-formed-p (rest e))))))))
      (dotimes (i 200)
        (let* ((a (mlx.symreg::random-expression 4 (evenp i) *vars* ops))
               (b (mlx.symreg::random-expression 3 nil *vars* ops))
               (k (random (sr:expression-size a))))
          (is (well-formed-p (mlx.symreg::mutate a *vars* ops)))
          (is (well-formed-p (mlx.symreg::crossover a b)))
          (is (equal a (mlx.symreg::replace-subexpression a k (mlx.symreg::subexpression a k)))))))))

(test stack-machine-matches-lisp-and-graphs
  (let* ((*random-state* (sb-ext:seed-random-state 5))
         (ops (mapcar #'mlx.symreg::find-operator sr:*default-operators*))
         (xs (samples 40))
         (exprs (append '((+ x0 (* 2.5 (sin x1))) (/ x0 0.0) (square (exp x1)) 3.0 x1
                          (- (/ x0 (+ x1 1.5)) (sqrt (cos x0))))
                        (loop repeat 100
                              collect (loop for e = (mlx.symreg::random-expression 4 nil *vars* ops)
                                            when (<= (sr:expression-size e) 25) return e))))
         (machine (sr:evaluate-expressions exprs xs :variables *vars*))
         (inputs (list (mx:from-lisp (mapcar #'first xs)) (mx:from-lisp (mapcar #'second xs)))))
    ;; the stack machine computes what each expression's own MLX graph does
    (loop for e in exprs for p from 0
          do (let ((graph (mx:to-lisp (mx:add (sr:expression->mlx e *vars* inputs) (mx:zeros (list (length xs)))))))
               (is (loop for i below (length xs) always (close-p (aref machine p i) (aref graph i)))
                   "~S differs from its graph" e)))
    ;; and, for fixed expressions, what the compiled Lisp function does
    (loop for e in (subseq exprs 0 6) for p from 0
          do (let ((f (sr:expression-function e *vars*)))
               (is (loop for row in xs for i from 0 always (close-p (aref machine p i) (apply f row)))
                   "~S differs from Lisp" e)))))

(defun fit-constants (exprs xs ys &key (scaling nil) (steps 200))
  (let* ((machine (mlx.symreg::%make-machine :variables *vars*
                                             :operators (mapcar #'mlx.symreg::find-operator sr:*default-operators*)
                                             :length 12 :stack-size 4 :scaling scaling))
         (x (mx:from-lisp (list (mapcar #'first xs) (mapcar #'second xs)) :dtype :float32))
         (y (mx:reshape (mx:from-lisp ys :dtype :float32) (list 1 -1)))
         (fit (lambda (&rest args) (apply #'mlx.symreg::machine-fit machine args))))
    (mlx.symreg::tune-and-score exprs machine (mlx.symreg::make-tuner machine 0.1) fit steps x y 1.0)))

(test constants-are-tuned-by-gradient
  (let* ((xs (samples 50))
         (ys (mapcar (lambda (r) (+ (* 3.7 (first r)) (sin (* 0.8 (second r))))) xs))
         (tuned (fit-constants '((+ (* 1.0 x0) (sin (* 1.0 x1))) (* 1.0 x0)) xs ys)))
    (destructuring-bind (good partial) tuned
      (is (< (sr:candidate-loss good) 1e-6))
      (destructuring-bind (plus (times a x) (sin (times* b y))) (sr:candidate-expression good)
        (declare (ignore plus times x sin times* y))
        (is (close-p 3.7 a 1e-3))
        (is (close-p 0.8 b 1e-3)))
      (is (> (sr:candidate-loss partial) 1e-2) "(* c x0) cannot fit the sine term"))))

(test linear-scaling
  (let* ((xs (samples 50))
         (ys (mapcar (lambda (r) (+ 2.0 (* -3.0 (square (first r))))) xs))
         (c (first (fit-constants '((square x0)) xs ys :scaling t :steps 1))))
    (is (< (sr:candidate-loss c) 1e-8))
    (destructuring-bind (plus a (times b e)) (sr:candidate-expression c)
      (is (eq '+ plus))
      (is (close-p 2.0 a))
      (is (eq '* times))
      (is (close-p -3.0 b))
      (is (equal '(square x0) e)))
    (is (equal '(square x0) (mlx.symreg::scaled-expression '(square x0) 0.0 1.0)))))

(test finds-a-formula
  (let* ((xs (samples 200 :seed 2))
         (ys (mapcar (lambda (r) (+ (square (first r)) (* 2.5 (sin (second r))))) xs)))
    (multiple-value-bind (best front)
        (sr:symbolic-regression xs ys :variables *vars* :seed 7 :population 300
                                      :generations 60 :stream nil)
      (is (< (sr:candidate-loss best) 1e-8) "best: ~S" best)
      (is (<= (sr:candidate-size best) 10))
      ;; the front is ordered by size, each better than all smaller ones
      (is (equal front (sort (copy-list front) #'< :key #'sr:candidate-size)))
      (is (loop for (a b) on front while b always (< (sr:candidate-loss b) (sr:candidate-loss a))))
      ;; and the formula found is a real Lisp function
      (let ((f (sr:expression-function (sr:candidate-expression best) *vars*)))
        (is (close-p (+ 4.0 (* 2.5 (sin 1.0))) (funcall f 2.0 1.0) 1e-3))))))
