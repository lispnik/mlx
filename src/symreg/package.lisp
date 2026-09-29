;;;; package.lisp -- mlx.symreg: symbolic regression by genetic programming

(defpackage :mlx.symreg
  (:use :cl)
  (:local-nicknames (:mx :mlx))
  (:export #:symbolic-regression
           #:*default-operators*
           #:operator-names
           ;; results
           #:candidate #:candidate-size #:candidate-loss #:candidate-expression
           ;; expressions
           #:square
           #:expression-function
           #:expression->mlx
           #:evaluate-expressions
           #:simplify-expression
           #:expression-size))
