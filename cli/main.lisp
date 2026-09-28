;;;; cli/main.lisp -- the mlx-cl command-line driver
;;;;
;;;; Build:  make cli        (or: sbcl --eval '(asdf:make "mlx/cli")')
;;;; Run:    bin/mlx-cl --help

(defpackage :mlx-cli
  (:use :cl)
  (:local-nicknames (:mx :mlx) (:random :mlx.random))
  (:export #:main #:top-level-command))

(in-package :mlx-cli)

;;; ------------------------------------------------------------------
;;; Shared options and helpers

(defun device-option ()
  (clingon:make-option :enum :long-name "device" :short-name #\d :key :device
                             :description "device to run on"
                             :items '(("gpu" . :gpu) ("cpu" . :cpu))
                             :initial-value "gpu"))

(defun dtype-option (&optional (default "float32"))
  (clingon:make-option :enum :long-name "dtype" :key :dtype
                             :description "element type"
                             :items (mapcar (lambda (d) (cons (string-downcase (symbol-name d)) d))
                                            (mx:dtypes))
                             :initial-value default))

(defun resolve-device (cmd)
  (let ((device (clingon:getopt cmd :device)))
    (when (and (eq device :gpu) (not (mx:metal-available-p)))
      (format *error-output* "warning: Metal is not available; using the CPU~%")
      (setf device :cpu))
    device))

(defun human-bytes (n)
  (loop for unit in '("B" "KiB" "MiB" "GiB" "TiB")
        for x = n then (/ x 1024.0)
        when (< x 1024) return (if (string= unit "B") (format nil "~D B" n) (format nil "~,1F ~A" x unit))
        finally (return (format nil "~D B" n))))

(defmacro with-mlx-errors (&body body)
  "Report errors as one line and exit non-zero instead of entering the debugger."
  `(handler-case (progn ,@body)
     (error (e)
       (format *error-output* "error: ~A~%" e)
       (uiop:quit 1))))

;;; ------------------------------------------------------------------
;;; info

(defun info-handler (cmd)
  (declare (ignore cmd))
  (with-mlx-errors
    (format t "MLX version     ~A~%" (mx:version))
    (format t "Default device  ~(~A~)~%" (mx:device-type (mx:default-device)))
    (format t "Metal           ~:[unavailable~;available~]~%" (mx:metal-available-p))
    (format t "CUDA            ~:[unavailable~;available~]~%" (mx:cuda-available-p))
    (dolist (type '(:cpu :gpu))
      (when (and (plusp (mx:device-count type)) (mx:device-available-p type))
        (format t "~%~:@(~A~) device:~%" type)
        (loop for (key value) on (mx:device-info type) by #'cddr
              do (format t "  ~(~28A~) ~A~%" key
                         (if (and (integerp value) (search "SIZE" (symbol-name key)))
                             (human-bytes value)
                             value)))))
    (format t "~%Memory: active ~A, cache ~A, peak ~A, limit ~A~%"
            (human-bytes (mx:active-memory)) (human-bytes (mx:cache-memory))
            (human-bytes (mx:peak-memory)) (human-bytes (mx:memory-limit)))))

(defun info-command ()
  (clingon:make-command :name "info"
                        :description "show MLX version, devices and memory"
                        :handler #'info-handler))

;;; ------------------------------------------------------------------
;;; eval

(defun eval-handler (cmd)
  (let ((args (clingon:command-arguments cmd))
        (device (resolve-device cmd)))
    (when (null args)
      (clingon:print-usage-and-exit cmd *error-output*))
    (with-mlx-errors
      (mx:with-device (device)
        (let ((*package* (find-package :mlx-user))
              (result nil))
          (with-input-from-string (in (format nil "~{~A~^ ~}" args))
            (loop for form = (read in nil in)
                  until (eq form in)
                  do (setf result (eval form))))
          (cond ((clingon:getopt cmd :dot)
                 (write-string (mx:export-to-dot result)))
                ((typep result 'mx:mlx-array)
                 (format t "~(~A~) (~{~D~^ ~})~%" (mx:dtype result) (mx:shape result))
                 (if (clingon:getopt cmd :lisp)
                     (format t "~S~%" (mx:to-lisp result :as :list))
                     (format t "~A~%" (array-text result))))
                (t (format t "~S~%" result))))))))

(defun array-text (array)
  "MLX's own rendering of ARRAY, e.g. \"array([1, 2], dtype=int32)\"."
  (let ((text (princ-to-string array)))
    ;; PRINT-OBJECT writes #<MLX-ARRAY dtype (shape)~%text>
    (subseq text (1+ (position #\Newline text)) (1- (length text)))))

(defun eval-command ()
  (clingon:make-command
   :name "eval"
   :description "evaluate Lisp forms in MLX-USER (nicknames MX, LINALG, FFT, RANDOM, FAST)"
   :usage "[options] FORM..."
   :options (list (device-option)
                  (clingon:make-option :flag :long-name "lisp" :short-name #\l :key :lisp
                                             :description "print the result as Lisp data")
                  (clingon:make-option :flag :long-name "dot" :key :dot
                                             :description "print the result's graph in DOT format"))
   :examples '(("Matrix product:" . "mlx-cl eval '(mx:matmul (mx:eye 2) (mx:ones (list 2 3)))'")
               ("As Lisp data:" . "mlx-cl eval -l '(mx:arange 5)'")
               ("Graphviz:" . "mlx-cl eval --dot '(mx:exp (mx:add (mx:ones (list 2)) 1))' | dot -Tpng > g.png"))
   :handler #'eval-handler))

;;; ------------------------------------------------------------------
;;; bench

(defun time-seconds (thunk)
  (let ((start (get-internal-real-time)))
    (funcall thunk)
    (/ (- (get-internal-real-time) start) internal-time-units-per-second)))

(defun bench-handler (cmd)
  (let ((n (clingon:getopt cmd :size))
        (iterations (clingon:getopt cmd :iterations))
        (dtype (clingon:getopt cmd :dtype))
        (device (resolve-device cmd)))
    (with-mlx-errors
      (mx:with-device (device)
        (let* ((a (random:normal :shape (list n n) :dtype dtype))
               (b (random:normal :shape (list n n) :dtype dtype)))
          (mx:eval a b)
          ;; warm up (kernel compilation, allocation)
          (mx:eval (mx:matmul a b))
          (let* ((seconds (time-seconds
                           (lambda ()
                             (dotimes (i iterations)
                               (mx:with-scope () (mx:eval (mx:matmul a b)))))))
                 (flops (* 2 n n n iterations)))
            (format t "matmul ~Dx~D ~(~A~) on ~(~A~): ~D iterations in ~,3F s~%"
                    n n dtype device iterations seconds)
            (format t "~,2F ms/iteration, ~,1F GFLOP/s~%"
                    (/ (* 1000 seconds) iterations)
                    (/ flops seconds 1d9))))))))

(defun bench-command ()
  (clingon:make-command
   :name "bench"
   :description "benchmark square matrix multiplication"
   :options (list (clingon:make-option :integer :long-name "size" :short-name #\n :key :size
                                             :description "matrix dimension" :initial-value 2048)
                  (clingon:make-option :integer :long-name "iterations" :short-name #\i
                                             :key :iterations :description "timed iterations"
                                             :initial-value 20)
                  (dtype-option)
                  (device-option))
   :handler #'bench-handler))

;;; ------------------------------------------------------------------
;;; inspect

(defun print-tensor-table (arrays)
  (let ((names (sort (loop for k being the hash-keys of arrays collect k) #'string<))
        (total 0))
    (format t "~&~40A ~10A ~20A ~12@A~%" "name" "dtype" "shape" "bytes")
    (dolist (name names)
      (let ((a (gethash name arrays)))
        (incf total (mx:nbytes a))
        (format t "~40A ~10A ~20A ~12@A~%" name (string-downcase (mx:dtype a))
                (format nil "~A" (mx:shape a)) (human-bytes (mx:nbytes a)))))
    (format t "~D tensors, ~A~%" (length names) (human-bytes total))))

(defun inspect-handler (cmd)
  (let ((files (clingon:command-arguments cmd)))
    (when (null files) (clingon:print-usage-and-exit cmd *error-output*))
    (with-mlx-errors
      (dolist (file files)
        (format t "~&== ~A~%" file)
        (let ((type (string-downcase (or (pathname-type (pathname file)) ""))))
          (cond ((string= type "safetensors")
                 (multiple-value-bind (arrays metadata) (mx:load file)
                   (print-tensor-table arrays)
                   (when (plusp (hash-table-count metadata))
                     (format t "metadata:~%")
                     (maphash (lambda (k v) (format t "  ~A: ~A~%" k v)) metadata))))
                ((string= type "gguf")
                 (print-tensor-table (mx:load file)))
                (t
                 (let ((a (mx:load file)))
                   (format t "~(~A~) ~A (~A)~%" (mx:dtype a) (mx:shape a) (human-bytes (mx:nbytes a)))
                   (when (clingon:getopt cmd :values)
                     (format t "~A~%" a))))))))))

(defun inspect-command ()
  (clingon:make-command
   :name "inspect"
   :description "list the tensors in .npy, .safetensors or .gguf files"
   :usage "[options] FILE..."
   :options (list (clingon:make-option :flag :long-name "values" :short-name #\v :key :values
                                             :description "print .npy values"))
   :handler #'inspect-handler))

;;; ------------------------------------------------------------------
;;; train: a small end-to-end autodiff demo

(defun train-handler (cmd)
  (let ((steps (clingon:getopt cmd :steps))
        (lr (float (/ (clingon:getopt cmd :lr-milli) 1000) 1f0))
        (n (clingon:getopt cmd :samples))
        (device (resolve-device cmd)))
    (with-mlx-errors
      (mx:with-device (device)
        (random:seed 0)
        ;; y = 3x0 - 2x1 + 0.5 + noise
        (let* ((true-w (mx:from-lisp '((3.0) (-2.0))))
               (x (random:normal :shape (list n 2)))
               (y (mx:add (mx:add (mx:matmul x true-w) 0.5)
                          (mx:multiply (random:normal :shape (list n 1)) 0.01)))
               (params (list :w (mx:zeros '(2 1)) :b (mx:zeros '(1))))
               (loss-fn (lambda (p)
                          (mx:mean (mx:square (mx:subtract (mx:add (mx:matmul x (getf p :w)) (getf p :b))
                                                           y)))))
               (step (mx:compile
                      (let ((vg (mx:value-and-grad loss-fn)))
                        (lambda (p)
                          (multiple-value-bind (loss grads) (funcall vg p)
                            (list loss (mx:tree-map (lambda (w g) (mx:subtract w (mx:multiply g lr)))
                                                    p grads))))))))
          (dotimes (i steps)
            (mx:with-scope ()
              (destructuring-bind (loss new-params) (funcall step params)
                (setf params (mx:keep new-params))
                (mx:eval loss params)
                (when (or (zerop (mod i (max 1 (floor steps 10)))) (= i (1- steps)))
                  (format t "step ~5D  loss ~,6F~%" i (mx:item loss))))))
          (format t "learned w = ~A, b = ~A  (true: (3 -2), 0.5)~%"
                  (mx:to-lisp (mx:flatten (getf params :w)) :as :list)
                  (mx:to-lisp (getf params :b) :as :list)))))))

(defun train-command ()
  (clingon:make-command
   :name "train"
   :description "fit a linear model with compiled value-and-grad (autodiff demo)"
   :options (list (clingon:make-option :integer :long-name "steps" :short-name #\s :key :steps
                                             :description "gradient steps" :initial-value 200)
                  (clingon:make-option :integer :long-name "lr-milli" :key :lr-milli
                                             :description "learning rate in thousandths"
                                             :initial-value 100)
                  (clingon:make-option :integer :long-name "samples" :key :samples
                                             :description "training samples" :initial-value 256)
                  (device-option))
   :handler #'train-handler))

;;; ------------------------------------------------------------------

(defun top-level-command ()
  (clingon:make-command
   :name "mlx-cl"
   :version #.(asdf:component-version (asdf:find-system "mlx"))
   :description "Common Lisp driver for Apple MLX"
   :authors '("mkennedy")
   :license "MIT"
   :handler (lambda (cmd) (clingon:print-usage-and-exit cmd t))
   :sub-commands (list (info-command) (eval-command) (bench-command)
                       (inspect-command) (train-command))))

(defun main ()
  (clingon:run (top-level-command)))
