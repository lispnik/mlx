;;;; tests/linalg-fft-random.lisp

(in-package :mlx-tests)

(def-suite :mlx.linalg :in :mlx)
(in-suite :mlx.linalg)

;; Most linalg routines run on the CPU in MLX.
(defmacro on-cpu (&body body) `(mx:with-device (:cpu) ,@body))

(test inverse-and-solve
  (on-cpu
    (let ((a (mx:from-lisp '((2.0 0.0) (0.0 4.0)))))
      (is (close-to (linalg:inv a) '((0.5 0) (0 0.25))))
      (is (close-to (linalg:solve a (mx:from-lisp '(2.0 4.0))) '(1 1)))
      (is (close-to (linalg:pinv a) '((0.5 0) (0 0.25))))
      (is (close-to (linalg:tri-inv a) '((0.5 0) (0 0.25))))
      (is (close-to (linalg:solve-triangular a (mx:from-lisp '(2.0 4.0))) '(1 1))))))

(test decompositions
  (on-cpu
    (let* ((a (mx:from-lisp '((4.0 2.0) (2.0 3.0)))))
      (let ((l (linalg:cholesky a)))
        (is (close-to (mx:matmul l (mx:transpose l)) '((4 2) (2 3)))))
      (multiple-value-bind (q r) (linalg:qr a)
        (is (close-to (mx:matmul q r) '((4 2) (2 3)))))
      (destructuring-bind (u s vt) (linalg:svd a)
        (is (close-to (mx:matmul (mx:multiply u s) vt) '((4 2) (2 3)))))
      (multiple-value-bind (w v) (linalg:eigh a)
        (is (= 2 (mx:size w)))
        (is (close-to (mx:matmul a v) (lisp (mx:multiply v w)))))
      (is (close-to (linalg:eigvalsh a) (lisp (mx:sort (linalg:eigvalsh a)))))
      (is (= 3 (length (linalg:lu a))))
      (multiple-value-bind (lu pivots) (linalg:lu-factor a)
        (is (equal '(2 2) (mx:shape lu)))
        (is (equal '(2) (mx:shape pivots)))))))

(test norms
  (on-cpu
    (let ((v (mx:from-lisp '(3.0 4.0)))
          (m (mx:from-lisp '((1.0 2.0) (3.0 4.0)))))
      (is (close-to (linalg:norm v) 5))
      (is (close-to (linalg:norm v :ord 1) 7))
      (is (close-to (linalg:norm v :ord :inf) 4))
      (is (close-to (linalg:norm m :ord "fro") (sqrt 30)))
      (is (close-to (linalg:norm m :axis 1) (list (sqrt 5) 5)))
      (is (close-to (linalg:cross (mx:from-lisp '(1.0 0.0 0.0)) (mx:from-lisp '(0.0 1.0 0.0)))
                    '(0 0 1))))))

(def-suite :mlx.fft :in :mlx)
(in-suite :mlx.fft)

(test fft-roundtrip
  (let* ((x (mx:from-lisp '(1.0 2.0 3.0 4.0)))
         (f (fft:fft x)))
    (is (eq :complex64 (mx:dtype f)))
    (is (approx= (mx:item (mx:ref f 0)) #c(10.0 0.0)))
    (is (close-to (mx:real (fft:ifft f)) '(1 2 3 4)))
    (is (equal '(3) (mx:shape (fft:rfft x))))
    (is (close-to (fft:irfft (fft:rfft x)) '(1 2 3 4)))
    (is (equal '(8) (mx:shape (fft:fft x :n 8))))))

(test fft-nd
  (let ((x (random:normal :shape '(4 6))))
    (is (equal '(4 6) (mx:shape (fft:fft2 x))))
    (is (equal '(4 4) (mx:shape (fft:rfft2 x))))
    (is (close-to (fft:irfft2 (fft:rfft2 x)) (lisp x) 1e-3))
    (is (close-to (mx:real (fft:ifftn (fft:fftn x))) (lisp x) 1e-3))
    (is (close-to (fft:ifftshift (fft:fftshift (mx:arange 5))) '(0 1 2 3 4)))
    (is (close-to (fft:fftfreq 4) '(0 0.25 -0.5 -0.25)))
    (is (close-to (fft:rfftfreq 4) '(0 0.25 0.5)))))

(def-suite :mlx.random :in :mlx)
(in-suite :mlx.random)

(test seeding-is-reproducible
  (random:seed 7)
  (let ((a (lisp (random:uniform :shape '(4)))))
    (random:seed 7)
    (is (equal a (lisp (random:uniform :shape '(4)))))))

(test explicit-keys
  (let ((key (random:key 0)))
    (is (equal (lisp (random:normal :shape '(3) :key key))
               (lisp (random:normal :shape '(3) :key key))))
    (is (equal '(3 2) (mx:shape (random:split key :num 3))))
    (multiple-value-bind (k1 k2) (random:split-pair key)
      (is (not (equal (lisp k1) (lisp k2)))))))

(test distributions
  (random:seed 1)
  (let ((u (random:uniform :shape '(1000) :low 2.0 :high 3.0)))
    (is (<= 2.0 (mx:item (mx:min u))))
    (is (> 3.0 (mx:item (mx:max u)))))
  (is (< (abs (mx:item (mx:mean (random:normal :shape '(10000))))) 0.05))
  (is (close-to (mx:mean (random:normal :shape '(10000) :loc 5.0)) 5 0.02))
  (let ((r (random:randint 0 10 :shape '(100))))
    (is (eq :int32 (mx:dtype r)))
    (is (<= 0 (mx:item (mx:min r))))
    (is (> 10 (mx:item (mx:max r)))))
  (is (eq :bool (mx:dtype (random:bernoulli :p 0.5 :shape '(10)))))
  (is (equal '(5) (mx:shape (random:truncated-normal -1 1 :shape '(5)))))
  (is (equal '(5) (mx:shape (random:gumbel :shape '(5)))))
  (is (equal '(5) (mx:shape (random:laplace :shape '(5)))))
  (is (equal '(4) (mx:shape (random:bits :shape '(4)))))
  (is (equal '(3 2) (mx:shape (mx:with-device (:cpu)
                                (random:multivariate-normal (mx:zeros '(2)) (mx:eye 2)
                                                            :shape '(3)))))))

(test categorical-and-permutation
  (let ((logits (mx:from-lisp '(-100.0 100.0 -100.0))))
    (is (= 1 (mx:item (random:categorical logits))))
    (is (equal '(5) (mx:shape (random:categorical logits :num-samples 5))))
    (is (equal '(2) (mx:shape (random:categorical (mx:broadcast-to logits '(2 3)) :shape '(2))))))
  (let ((p (random:permutation 10)))
    (is (equal (loop for i below 10 collect i) (lisp (mx:sort p)))))
  (is (equal '(4 5 6) (lisp (mx:sort (random:permutation (mx:from-lisp '(6 4 5))))))))
