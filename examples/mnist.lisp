;;;; examples/mnist.lisp -- handwritten digit classification with mlx.nn
;;;;
;;;; Trains a multi-layer perceptron, then a small convolutional network,
;;;; on MNIST and reports test accuracy after each epoch.  Run from the
;;;; project root:
;;;;
;;;;   sbcl --load examples/mnist.lisp --eval '(mnist:main)' --quit
;;;;
;;;; The dataset (~11 MB) is downloaded to ~/.cache/mlx-cl/datasets/mnist/.

(asdf:load-system "mlx")

(defpackage :mnist
  (:use :cl)
  (:local-nicknames (:mx :mlx) (:nn :mlx.nn) (:optim :mlx.optimizers) (:random :mlx.random))
  (:export #:main #:load-mnist #:train))

(in-package :mnist)

;;; ------------------------------------------------------------------
;;; Data

(defparameter *mirror* "https://ossci-datasets.s3.amazonaws.com/mnist/")

(defun data-dir ()
  (merge-pathnames ".cache/mlx-cl/datasets/mnist/" (user-homedir-pathname)))

(defun fetch (name)
  "The decompressed IDX file NAME, downloading it on first use."
  (let ((file (merge-pathnames name (data-dir))))
    (unless (probe-file file)
      (ensure-directories-exist file)
      (format t "~&Downloading ~A~%" name)
      (uiop:run-program (list "sh" "-c" (format nil "curl -fsSL '~A~A.gz' | gunzip > '~A'"
                                                *mirror* name (uiop:native-namestring file)))
                        :error-output t))
    file))

(defun read-idx (name)
  "An IDX file as (values octets dimensions)."
  (with-open-file (in (fetch name) :element-type '(unsigned-byte 8))
    (flet ((u32 () (let ((v 0)) (dotimes (i 4 v) (setf v (+ (* v 256) (read-byte in)))))))
      (let* ((magic (u32))
             (dims (loop repeat (ldb (byte 8 0) magic) collect (u32)))
             (data (make-array (reduce #'* dims) :element-type '(unsigned-byte 8))))
        (read-sequence data in)
        (values data dims)))))

(defun load-split (images labels)
  "Returns (values images labels): float32 (N 28 28 1) scaled to [0,1], and int32 (N)."
  (multiple-value-bind (pixels dims) (read-idx images)
    (values (mx:divide (mx:reshape (mx:from-lisp pixels :dtype :float32)
                                   (append dims '(1)))
                       255.0)
            (mx:from-lisp (read-idx labels) :dtype :int32))))

(defun load-mnist ()
  "Returns (values train-x train-y test-x test-y)."
  (multiple-value-bind (train-x train-y) (load-split "train-images-idx3-ubyte" "train-labels-idx1-ubyte")
    (multiple-value-bind (test-x test-y) (load-split "t10k-images-idx3-ubyte" "t10k-labels-idx1-ubyte")
      (values train-x train-y test-x test-y))))

;;; ------------------------------------------------------------------
;;; Models

(defun make-mlp ()
  (nn:sequential (lambda (x) (mx:reshape x (list (mx:dim x 0) -1)))
                 (nn:linear 784 256) #'nn:relu
                 (nn:linear 256 256) #'nn:relu
                 (nn:linear 256 10)))

(defun make-cnn ()
  (nn:sequential (nn:conv2d 1 16 3 :padding 1) #'nn:relu (nn:max-pool-2d 2)   ; 14x14
                 (nn:conv2d 16 32 3 :padding 1) #'nn:relu (nn:max-pool-2d 2)  ; 7x7
                 (lambda (x) (mx:reshape x (list (mx:dim x 0) -1)))
                 (nn:linear (* 7 7 32) 128) #'nn:relu
                 (nn:dropout 0.25)
                 (nn:linear 128 10)))

;;; ------------------------------------------------------------------
;;; Training

(defun accuracy (model x y &key (batch 1000))
  (nn:train-mode model nil)
  (prog1 (/ (loop for start from 0 below (mx:dim x 0) by batch
                  sum (mx:with-scope ()
                        (let ((end (min (mx:dim x 0) (+ start batch))))
                          (mx:item (mx:sum (mx:equal (mx:argmax (funcall model (mx:ref x (list start end))) :axis 1)
                                                     (mx:ref y (list start end))))))))
            (mx:dim x 0))
    (nn:train-mode model t)))

(defun train (model name train-x train-y test-x test-y &key (epochs 3) (batch 128) (learning-rate 1e-3))
  (let* ((opt (optim:adamw learning-rate))
         (step (nn:value-and-grad model (lambda (x y)
                                          (nn:cross-entropy (funcall model x) y :reduction :mean))))
         (n (mx:dim train-x 0)))
    (format t "~&~A: ~:D parameters~%" name (nn:parameter-count model))
    (dotimes (epoch epochs)
      (let ((order (random:permutation n))
            (start-time (get-internal-real-time))
            (loss-sum 0) (batches 0))
        (loop for start from 0 below n by batch
              do (mx:with-scope ()
                   (let* ((idx (mx:ref order (list start (min n (+ start batch)))))
                          (x (mx:take train-x idx :axis 0))
                          (y (mx:take train-y idx :axis 0)))
                     (multiple-value-bind (loss grads) (funcall step x y)
                       (optim:update opt model grads)
                       (mx:eval loss (nn:parameters model) (optim:state opt))
                       (incf loss-sum (mx:item loss))
                       (incf batches)))))
        (format t "~&  epoch ~D  loss ~,4F  test accuracy ~,2F%  (~,1Fs)~%"
                (1+ epoch) (/ loss-sum batches) (* 100 (accuracy model test-x test-y))
                (/ (- (get-internal-real-time) start-time) internal-time-units-per-second))))
    model))

(defun main (&key (epochs 3))
  (random:seed 0)
  (multiple-value-bind (train-x train-y test-x test-y) (load-mnist)
    (format t "~&MNIST: ~:D training and ~:D test images~%" (mx:dim train-x 0) (mx:dim test-x 0))
    (train (make-mlp) "MLP" train-x train-y test-x test-y :epochs epochs)
    (train (make-cnn) "CNN" train-x train-y test-x test-y :epochs epochs)))
