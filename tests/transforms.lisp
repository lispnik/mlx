;;;; tests/transforms.lisp -- grad, vjp, jvp, vmap, compile, custom functions

(in-package :mlx-tests)

(def-suite :mlx.transforms :in :mlx)
(in-suite :mlx.transforms)

(defun sum-of-squares (x) (mx:sum (mx:square x)))

(test grad-basic
  (is (close-to (funcall (mx:grad #'sum-of-squares) 3.0) 6))
  (is (close-to (funcall (mx:grad #'sum-of-squares) (mx:from-lisp '(1.0 2.0))) '(2 4)))
  (is (close-to (funcall (mx:grad (lambda (x) (mx:sum (mx:sin x)))) (mx:scalar 0.0)) 1)))

(test grad-argnums
  (let ((f (lambda (x y) (mx:sum (mx:multiply x y)))))
    (is (close-to (funcall (mx:grad f :argnums 1) (mx:scalar 2.0) (mx:scalar 5.0)) 2))
    (destructuring-bind (gx gy) (funcall (mx:grad f :argnums '(0 1)) (mx:scalar 2.0) (mx:scalar 5.0))
      (is (close-to gx 5))
      (is (close-to gy 2))))
  (signals error (funcall (mx:grad #'sum-of-squares :argnums 3) (mx:scalar 1.0))))

(test value-and-grad-trees
  (let ((loss (lambda (params x)
                (mx:sum (mx:add (mx:multiply (getf params :w) x) (getf params :b))))))
    (multiple-value-bind (value grads)
        (funcall (mx:value-and-grad loss)
                 (list :w (mx:from-lisp '(1.0 2.0)) :b (mx:scalar 0.5))
                 (mx:from-lisp '(3.0 4.0)))
      (is (close-to value 12))
      (is (close-to (getf grads :w) '(3 4)))
      (is (close-to (getf grads :b) 2))))
  ;; hash-table parameters
  (let ((params (make-hash-table :test 'equal)))
    (setf (gethash "w" params) (mx:scalar 3.0))
    (let ((g (funcall (mx:grad (lambda (p) (mx:square (gethash "w" p)))) params)))
      (is (close-to (gethash "w" g) 6)))))

(test value-and-grad-aux
  (multiple-value-bind (value grads)
      (funcall (mx:value-and-grad (lambda (x) (list (mx:sum (mx:square x)) (mx:multiply x 10))))
               (mx:from-lisp '(1.0 2.0)))
    (is (close-to (first value) 5))
    (is (close-to (second value) '(10 20)))
    (is (close-to grads '(2 4)))))

(test callback-errors-propagate
  (signals simple-error (funcall (mx:grad (lambda (x) (declare (ignore x)) (error "boom"))) 1.0))
  ;; the system is still healthy afterwards
  (is (close-to (funcall (mx:grad #'sum-of-squares) 1.0) 2)))

(test vjp-and-jvp
  (multiple-value-bind (outs vjps)
      (mx:vjp (lambda (x) (mx:multiply x x)) (list (mx:from-lisp '(1.0 2.0))) (list (mx:ones '(2))))
    (is (close-to (first outs) '(1 4)))
    (is (close-to (first vjps) '(2 4))))
  (multiple-value-bind (outs jvps)
      (mx:jvp (lambda (x y) (mx:multiply x y)) (list (mx:scalar 2.0) (mx:scalar 3.0))
              (list (mx:scalar 1.0) (mx:scalar 0.0)))
    (is (close-to (first outs) 6))
    (is (close-to (first jvps) 3))))

(test vmap
  (let ((f (mx:vmap (lambda (x y) (mx:add x y)) :in-axes '(0 nil))))
    (is (close-to (funcall f (mx:from-lisp '((1 2) (3 4))) (mx:from-lisp '(10 20)))
                  '((11 22) (13 24)))))
  (let ((dot (mx:vmap (lambda (a b) (mx:sum (mx:multiply a b))))))
    (is (close-to (funcall dot (mx:from-lisp '((1 2) (3 4))) (mx:from-lisp '((1 1) (2 2)))) '(3 14))))
  (is (close-to (funcall (mx:vmap #'mx:transpose :in-axes 1 :out-axes 0) (mx:reshape (mx:arange 8) '(2 2 2)))
                (lisp (mx:transpose (mx:reshape (mx:arange 8) '(2 2 2)) :axes '(1 2 0))))))

(test compile
  (let* ((calls 0)
         (f (mx:compile (lambda (x y) (incf calls) (mx:add (mx:multiply x y) 1.0)))))
    (is (close-to (funcall f (mx:from-lisp '(1.0 2.0)) (mx:from-lisp '(3.0 4.0))) '(4 9)))
    (is (close-to (funcall f (mx:from-lisp '(2.0 2.0)) (mx:from-lisp '(3.0 4.0))) '(7 9)))
    (is (= 1 calls) "traced once, then cached")
    ;; a new shape retraces
    (is (close-to (funcall f (mx:ones '(3)) (mx:ones '(3))) '(2 2 2)))
    (is (= 2 calls)))
  (let ((g (mx:compile (lambda (x) (list (mx:sin x) (mx:cos x))))))
    (destructuring-bind (s c) (funcall g (mx:scalar 0.0))
      (is (close-to s 0))
      (is (close-to c 1))))
  (let ((h (mx:compile (lambda (x) (mx:multiply x 2)) :shapeless t)))
    (is (close-to (funcall h (mx:ones '(2))) '(2 2)))
    (is (close-to (funcall h (mx:ones '(3))) '(2 2 2))))
  ;; multiple values survive compilation
  (multiple-value-bind (v g) (funcall (mx:compile (mx:value-and-grad #'sum-of-squares))
                                      (mx:from-lisp '(1.0 2.0)))
    (is (close-to v 5))
    (is (close-to g '(2 4))))
  ;; gradients of compiled functions
  (is (close-to (funcall (mx:grad (mx:compile #'sum-of-squares)) (mx:from-lisp '(1.0 3.0))) '(2 6))))

(test functions-made-in-a-scope-outlive-it
  ;; handles owned by returned functions must not be freed by WITH-SCOPE
  (let ((f (mx:with-scope () (mx:compile (lambda (x) (mx:add x 1)))))
        (g (mx:with-scope () (mx:checkpoint (lambda (x) (mx:multiply x 2)))))
        (h (mx:with-scope () (mx:custom-vjp (lambda (x) (mx:sin x))
                                            (lambda (p c o) (declare (ignore p o)) (list (first c)))))))
    (is (close-to (funcall f (mx:scalar 1.0)) 2))
    (is (close-to (funcall g (mx:scalar 1.0)) 2))
    (is (close-to (funcall h (mx:scalar 0.0)) 0))))

(test compile-control
  (mx:disable-compile)
  (is (close-to (funcall (mx:compile #'sum-of-squares) (mx:scalar 2.0)) 4))
  (mx:enable-compile)
  (is (eq :no-fuse (mx:set-compile-mode :no-fuse)))
  (mx:set-compile-mode :enabled)
  (is (null (mx:clear-compile-cache))))

(test checkpoint
  (let ((f (mx:checkpoint (lambda (x) (mx:sin x)))))
    (is (close-to (funcall f (mx:scalar 0.0)) 0))
    (is (close-to (funcall (mx:grad (lambda (x) (mx:sum (funcall f x)))) (mx:scalar 0.0)) 1))))

(test custom-functions
  (let ((f (mx:custom-vjp (lambda (x) (mx:sin x))
                          (lambda (primals cotangents outputs)
                            (declare (ignore primals outputs))
                            (list (mx:multiply (first cotangents) 10.0))))))
    (is (close-to (funcall f (mx:scalar 0.0)) 0))
    (is (close-to (funcall (mx:grad (lambda (x) (mx:sum (funcall f x)))) (mx:scalar 0.5)) 10)))
  (let ((g (mx:custom-function (lambda (x) (mx:multiply x 2))
                               :jvp (lambda (primals tangents argnums)
                                      (declare (ignore primals argnums))
                                      (list (mx:multiply (first tangents) 100))))))
    (multiple-value-bind (outs jvps) (mx:jvp g (list (mx:scalar 1.0)) (list (mx:scalar 1.0)))
      (is (close-to (first outs) 2))
      (is (close-to (first jvps) 100)))))

(test training-loop
  ;; fit y = 2x + 1 with plain gradient descent
  (random:seed 0)
  (let* ((x (random:normal :shape '(64 1)))
         (y (mx:add (mx:multiply x 2.0) 1.0))
         (params (list :w (mx:zeros '(1 1)) :b (mx:zeros '(1))))
         (loss (lambda (p)
                 (mx:mean (mx:square (mx:subtract (mx:add (mx:matmul x (getf p :w)) (getf p :b)) y)))))
         (step (mx:value-and-grad loss)))
    (dotimes (i 200)
      (mx:with-scope ()
        (multiple-value-bind (l g) (funcall step params)
          (declare (ignore l))
          (setf params (mx:keep (mx:tree-map (lambda (p d) (mx:subtract p (mx:multiply d 0.1)))
                                             params g)))
          (mx:eval params))))
    (is (close-to (getf params :w) '((2)) 1e-2))
    (is (close-to (getf params :b) '(1) 1e-2))))
