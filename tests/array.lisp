;;;; tests/array.lisp -- creation, conversion, properties, indexing, lifetime

(in-package :mlx-tests)

(def-suite :mlx.array :in :mlx)
(in-suite :mlx.array)

(test scalars
  (is (eql 3 (mx:item (mx:scalar 3))))
  (is (eq :int32 (mx:dtype (mx:scalar 3))))
  (is (eq :float32 (mx:dtype (mx:scalar 2.5d0))))
  (is (= 2.5 (mx:item (mx:scalar 2.5))))
  (is (eq :bool (mx:dtype (mx:scalar t))))
  (is (eq t (mx:item (mx:scalar t))))
  (is (eq :complex64 (mx:dtype (mx:scalar #c(1 2)))))
  (is (= #c(1.0 2.0) (mx:item (mx:scalar #c(1 2)))))
  (is (eq :int64 (mx:dtype (mx:scalar (expt 2 40)))))
  (is (= (expt 2 40) (mx:item (mx:scalar (expt 2 40))))))

(test from-nested-lists
  (let ((a (mx:from-lisp '((1 2 3) (4 5 6)))))
    (is (equal '(2 3) (mx:shape a)))
    (is (eq :int32 (mx:dtype a)))
    (is (= 2 (mx:ndim a)))
    (is (= 6 (mx:size a)))
    (is (= 4 (mx:itemsize a)))
    (is (= 24 (mx:nbytes a)))
    (is (= 3 (mx:dim a -1)))
    (is (equal '((1 2 3) (4 5 6)) (lisp a))))
  (is (eq :float32 (mx:dtype (mx:from-lisp '(1 2.5)))))
  (is (eq :bool (mx:dtype (mx:from-lisp '(t nil t)))))
  (is (equal '(t nil t) (lisp (mx:from-lisp '(t nil t)))))
  (is (equal '(0) (mx:shape (mx:from-lisp '()))))
  (signals error (mx:from-lisp '((1 2) (3)))))

(test from-lisp-arrays
  (let ((a (mx:from-lisp (make-array '(2 2) :element-type 'single-float
                                            :initial-contents '((1.0 2.0) (3.0 4.0))))))
    (is (eq :float32 (mx:dtype a)))
    (is (equalp #2A((1.0 2.0) (3.0 4.0)) (mx:to-lisp a))))
  (is (eq :float64 (mx:dtype (mx:from-lisp (make-array 3 :element-type 'double-float
                                                         :initial-element 1d0)))))
  (is (eq :uint8 (mx:dtype (mx:from-lisp (make-array 3 :element-type '(unsigned-byte 8)
                                                       :initial-element 7)))))
  (is (eq :int16 (mx:dtype (mx:from-lisp (make-array 3 :element-type '(signed-byte 16)
                                                       :initial-element -7)))))
  (is (equalp #(1 2 3) (mx:to-lisp (mx:from-lisp #(1 2 3)))))
  (is (equal '((1 2) (3 4)) (lisp (mx:from-lisp #(#(1 2) #(3 4))))))
  (is (equalp #(#c(1.0 0.0) #c(0.0 1.0))
              (mx:to-lisp (mx:from-lisp (make-array 2 :element-type '(complex single-float)
                                                      :initial-contents '(#c(1.0 0.0) #c(0.0 1.0))))))))

(test explicit-dtypes
  (dolist (dtype '(:bool :uint8 :uint16 :uint32 :uint64 :int8 :int16 :int32 :int64
                   :float16 :float32 :bfloat16 :complex64))
    (let ((a (mx:from-lisp '(0 1 1) :dtype dtype)))
      (is (eq dtype (mx:dtype a)) "dtype ~S" dtype)
      (is (= 3 (mx:size a)))
      (is (= (mx:dtype-size dtype) (mx:itemsize a)))))
  (is (approx= '(0.5 1.5) (lisp (mx:from-lisp '(0.5 1.5) :dtype :float16))))
  (is (approx= '(0.5 1.5) (lisp (mx:from-lisp '(0.5 1.5) :dtype :bfloat16))))
  (is (= 14 (length (mx:dtypes))))
  (signals error (mx:from-lisp '(1.5) :dtype :int32))
  (signals error (mx:from-lisp '(1) :dtype :nonsense)))

(test to-lisp-specialized
  (is (typep (mx:to-lisp (mx:ones '(3))) '(simple-array single-float (3))))
  (is (typep (mx:to-lisp (mx:arange 3)) '(simple-array (signed-byte 32) (3))))
  (is (typep (mx:to-lisp (mx:arange 3 :dtype :uint8)) '(simple-array (unsigned-byte 8) (3))))
  ;; non-contiguous (transposed) and broadcast arrays read back in logical order
  (let ((a (mx:reshape (mx:arange 6) '(2 3))))
    (is (equal '((0 3) (1 4) (2 5)) (lisp (mx:transpose a))))
    (is (equal '((0 1 2) (0 1 2)) (lisp (mx:broadcast-to (mx:arange 3) '(2 3)))))))

(test ensure-array-identity
  (let ((a (mx:ones '(2))))
    (is (eq a (mx:ensure-array a)))
    (is (not (eq a (mx:ensure-array a :dtype :int32))))
    (is (eq :int32 (mx:dtype (mx:ensure-array a :dtype :int32))))))

(test item-errors
  (signals error (mx:item (mx:ones '(2)))))

(test printing
  (let ((s (princ-to-string (mx:from-lisp '(1 2)))))
    (is (search "int32" s))
    (is (search "array([1, 2]" s)))
  (let ((a (mx:ones '(2))))
    (mx:free a)
    (is (search "freed" (princ-to-string a)))))

(test indexing
  (let ((a (mx:reshape (mx:arange 12) '(3 4))))
    (is (equal '(4 5 6 7) (lisp (mx:ref a 1))))
    (is (equal '(8 9 10 11) (lisp (mx:ref a -1))))
    (is (= 6 (mx:item (mx:ref a 1 2))))
    (is (equal '(2 6 10) (lisp (mx:ref a t 2))))
    (is (equal '(2 6 10) (lisp (mx:ref a :all 2))))
    (is (equal '((4 5 6 7) (8 9 10 11)) (lisp (mx:ref a '(1 nil)))))
    (is (equal '((1 3) (9 11)) (lisp (mx:ref a '(0 nil 2) '(1 nil 2)))))
    (is (equal '(8 4 0) (lisp (mx:ref a '(nil nil -1) 0))))
    (is (equal '(3 2 1 0) (lisp (mx:ref a 0 '(nil nil -1)))))
    (is (equal '(1 4) (mx:shape (mx:ref a :newaxis 0))))
    (is (equal '(3 1 4) (mx:shape (mx:ref a t :newaxis))))
    (is (equal '((8 9 10 11) (0 1 2 3)) (lisp (mx:ref a #(2 0)))))
    (is (equal '(0 4) (mx:shape (mx:ref a '(2 0)))) "a list is a slice, here empty")
    (is (equal '((0 2) (4 6) (8 10)) (lisp (mx:ref a t (mx:from-lisp '(0 2))))))
    (signals error (mx:ref a 5))
    (signals error (mx:ref a 0 0 0))))

(test setf-indexing
  (let ((a (mx:zeros '(2 3))))
    (setf (mx:ref a 0) 1)
    (is (equal '((1.0 1.0 1.0) (0.0 0.0 0.0)) (lisp a)))
    (setf (mx:ref a t 2) (mx:from-lisp '(7 8)))
    (is (equal '((1.0 1.0 7.0) (0.0 0.0 8.0)) (lisp a)))
    (setf (mx:ref a 1 '(0 2)) (mx:from-lisp '(5 6)))
    (is (equal '((1.0 1.0 7.0) (5.0 6.0 8.0)) (lisp a)))
    (is (eq :float32 (mx:dtype a)))))

(test explicit-free
  (let ((a (mx:ones '(2))))
    (is (not (mx:freed-p a)))
    (mx:free a)
    (is (mx:freed-p a))
    (mx:free a)                         ; idempotent
    (signals error (mx:shape a))
    (signals error (mx:add a 1))))

(test with-scope
  (let (inner kept result)
    (setf result (mx:with-scope ()
                   (setf inner (mx:ones '(3)))
                   (setf kept (mx:keep (mx:zeros '(3))))
                   (mx:add inner 1)))
    (is (mx:freed-p inner))
    (is (not (mx:freed-p kept)))
    (is (not (mx:freed-p result)))
    (is (equal '(2.0 2.0 2.0) (lisp result))))
  ;; survivors of an inner scope are handed to the outer one
  (let (mid)
    (mx:with-scope ()
      (setf mid (mx:with-scope () (mx:ones '(2))))
      (is (not (mx:freed-p mid))))
    (is (mx:freed-p mid)))
  ;; trees of results survive
  (let ((pair (mx:with-scope () (list (mx:ones '(1)) (vector (mx:zeros '(1)))))))
    (is (not (mx:freed-p (first pair))))
    (is (not (mx:freed-p (aref (second pair) 0))))))

(test eval-and-async
  (let ((a (mx:add (mx:ones '(4)) 1)))
    (is (eq a (mx:eval a)))
    (is (eq a (mx:async-eval a)))
    (mx:eval (list a (mx:ones '(2))))
    (is (equal '(2.0 2.0 2.0 2.0) (lisp a)))))

(test trees
  (multiple-value-bind (leaves structure)
      (mx:tree-flatten (list :w (mx:ones '(2)) :b 3 :name "x" :v (vector 1 2)))
    (is (= 4 (length leaves)))
    (let ((back (mx:tree-unflatten structure '(a b c d))))
      (is (equalp '(:w a :b b :name "x" :v #(c d)) back))))
  (let* ((h (make-hash-table :test 'equal)))
    (setf (gethash "k" h) 1)
    (let ((mapped (mx:tree-map (lambda (x) (* x 10)) (list h 2))))
      (is (= 10 (gethash "k" (first mapped))))
      (is (= 20 (second mapped))))))
