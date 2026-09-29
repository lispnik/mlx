;;;; tests/defnet.lisp -- nn:defnet: shape inference, compile-time and
;;;; runtime shape errors, composition, training

(in-package :mlx-tests)

(def-suite :mlx.defnet :in :mlx)
(in-suite :mlx.defnet)

(nn:defnet test-cnn ((x (batch 12 12 1)) &key (classes 10))
  "A small CNN."
  (-> x
      (conv2d 4 3 :padding 1) relu (max-pool-2d 2)   ; (batch 6 6 4)
      (conv2d 8 3) relu                              ; (batch 4 4 8)
      flatten                                        ; (batch 128)
      (linear 16) relu (dropout 0.1)
      (linear classes)))

(nn:defnet test-block ((x (b s dims)) &key dims (heads 2) (hidden 32))
  (-> x
      (residual (layer-norm) (attention heads :mask :causal))
      (residual (layer-norm) (linear hidden) gelu (linear dims))))

(nn:defnet test-lm ((tokens (b s)) &key (vocab 50) (dims 16))
  (-> tokens (embedding vocab dims) (repeat 2 (test-block)) (rms-norm) (linear vocab)))

(nn:defnet test-two ((a (b n)) (c (b n)))
  (let* ((sum (+ a c))
         (both (concat -1 sum (* 2 a))))
    (-> both (reshape b 2 n) (mean 1))))

(test defnet-infers-sizes-and-shapes
  (let ((cnn (test-cnn)))
    ;; (linear 16) got its 128 inputs from the inferred shape
    (is (equal '(16 128) (mx:shape (nn:child (nn:child cnn "linear_0") "weight"))))
    (is (equal '(5 10) (mx:shape (funcall cnn (mx:zeros '(5 12 12 1))))))
    (is (equal '(2 3) (mx:shape (funcall (test-cnn :classes 3) (mx:zeros '(2 12 12 1)))))))
  (let ((lm (test-lm)))
    (is (equal '(2 7 50) (mx:shape (funcall lm (mx:zeros '(2 7) :dtype :int32)))))
    ;; the blocks' DIMS was inferred from the shape they were applied to
    (is (= 16 (slot-value (nn:child lm "test_block_0") 'dims))))
  (is (equal '((2.0 3.0)) (lisp (funcall (test-two) (mx:from-lisp '((1.0 1.0))) (mx:from-lisp '((1.0 3.0))))))))

(test defnet-summary-matches-the-module
  (let* ((text (with-output-to-string (s) (nn:net-summary 'test-cnn s)))
         (cnn (test-cnn)))
    (is (search "(batch 128)" text))
    (is (search (format nil "~:D" (nn:parameter-count cnn)) text))))

(defmacro shape-error-message (form)
  `(handler-case (progn (macroexpand-1 ',form) nil)
     (nn:shape-error (e) (princ-to-string e))))

(test defnet-rejects-bad-shapes-at-compile-time
  (flet ((rejects (message text)
           (is (and text (search message text)) "expected ~S in ~S" message text)))
    (rejects "residual branch must keep the shape (b 5), but it gives (b 3)"
             (shape-error-message (nn:defnet bad ((x (b 10))) (-> x (linear 5) (residual (linear 3))))))
    (rejects "7 heads do not divide the width 60"
             (shape-error-message (nn:defnet bad ((x (b s 60))) (-> x (attention 7)))))
    (rejects "matmul inner dimensions differ"
             (shape-error-message (nn:defnet bad ((a (b 3 4)) (c (b 5 6))) (matmul a c))))
    (rejects "a window of 5 does not fit an input of 4"
             (shape-error-message (nn:defnet bad ((x (b 4 4 1))) (-> x (conv2d 8 5)))))
    (rejects "make s a &key hyperparameter"
             (shape-error-message (nn:defnet bad ((x (b s))) (-> x (linear 3)))))
    (rejects "cannot reshape (b 6)"
             (shape-error-message (nn:defnet bad ((x (b 6))) (-> x (reshape b 4 -1)))))
    (rejects "cannot broadcast (b 6) with (b 5)"
             (shape-error-message (nn:defnet bad ((x (b 6))) (+ x (-> x (linear 5))))))
    (rejects "write (-> x (linear 3))"
             (shape-error-message (nn:defnet bad ((x (b 6))) (linear x 3))))
    (rejects "test-block expects rank 3"
             (shape-error-message (nn:defnet bad ((x (b 6))) (-> x (test-block)))))
    (rejects "unknown stage"
             (shape-error-message (nn:defnet bad ((x (b 6))) (-> x (frobnicate 3)))))))

(test defnet-checks-inputs-at-run-time
  (signals nn:shape-error (funcall (test-cnn) (mx:zeros '(5 12 12 3))))
  (signals nn:shape-error (funcall (test-lm) (mx:zeros '(2 7 3) :dtype :int32)))
  ;; runtime dimensions must agree across inputs
  (signals nn:shape-error (funcall (test-two) (mx:zeros '(2 3)) (mx:zeros '(2 4))))
  (signals nn:shape-error (funcall (test-two) (mx:zeros '(2 3)) (mx:zeros '(1 3)))))

(test defnet-modules-train
  (random:seed 0)
  (let* ((model (test-cnn :classes 2))
         (x (random:normal :shape '(16 12 12 1)))
         ;; label: is the image's mean positive?
         (y (mx:astype (mx:greater (mx:mean x :axis '(1 2 3)) 0) :int32))
         (opt (optim:adam 1e-2))
         (loss (lambda () (nn:cross-entropy (funcall model x) y :reduction :mean)))
         (step (nn:value-and-grad model loss))
         (initial (mx:item (funcall loss))))
    (nn:train-mode model nil)
    (dotimes (i 60)
      (mx:with-scope ()
        (optim:update opt model (nth-value 1 (funcall step)))
        (mx:eval (nn:parameters model))))
    (is (< (mx:item (funcall loss)) (* 0.5 initial)))))
