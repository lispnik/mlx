;;;; examples/transformer.lisp -- a character-level GPT trained from scratch
;;;;
;;;; Trains a small decoder-only transformer on Tiny Shakespeare, then
;;;; samples text from it.  Run from the project root:
;;;;
;;;;   sbcl --load examples/transformer.lisp --eval '(char-gpt:main)' --quit
;;;;
;;;; The text (~1 MB) is downloaded to ~/.cache/mlx-cl/datasets/.  With the
;;;; defaults (4 layers, 256 dims, 1500 steps) the validation loss falls
;;;; from ~4.2 to ~1.6 nats per character in a few minutes on an M-series
;;;; GPU, and samples look Shakespeare-shaped.

(asdf:load-system "mlx")

(defpackage :char-gpt
  (:use :cl)
  (:local-nicknames (:mx :mlx) (:nn :mlx.nn) (:optim :mlx.optimizers) (:random :mlx.random))
  (:export #:main #:train #:sample))

(in-package :char-gpt)

;;; ------------------------------------------------------------------
;;; Data

(defparameter *url* "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

(defun load-text ()
  (let ((file (merge-pathnames ".cache/mlx-cl/datasets/tinyshakespeare.txt" (user-homedir-pathname))))
    (unless (probe-file file)
      (ensure-directories-exist file)
      (format t "~&Downloading Tiny Shakespeare~%")
      (uiop:run-program (list "curl" "-fsSL" "-o" (uiop:native-namestring file) *url*)))
    (uiop:read-file-string file)))

(defstruct corpus chars index train valid)

(defun make-dataset (text)
  "Character vocabulary and the text as int32 arrays (90% train, 10% valid)."
  (let* ((chars (sort (remove-duplicates text) #'char<))
         (index (let ((h (make-hash-table))) (loop for c across chars for i from 0 do (setf (gethash c h) i)) h))
         (ids (map '(simple-array (signed-byte 32) (*)) (lambda (c) (gethash c index)) text))
         (split (floor (* 0.9 (length ids)))))
    (make-corpus :chars chars :index index
                 :train (mx:from-lisp (subseq ids 0 split))
                 :valid (mx:from-lisp (subseq ids split)))))

(defun batch (data batch-size context)
  "Random windows: inputs (B, T) and next-character targets (B, T)."
  (let* ((starts (random:randint 0 (- (mx:size data) context 1) :shape (list batch-size 1)))
         (positions (mx:add starts (mx:arange (1+ context))))
         (windows (mx:take data positions)))
    (values (mx:ref windows t (list 0 context))
            (mx:ref windows t (list 1 nil)))))

;;; ------------------------------------------------------------------
;;; Model: pre-norm transformer blocks with causal self-attention

(nn:defmodule gpt-block () ())

(defun make-block (dims heads)
  (let ((m (make-instance 'gpt-block)))
    (nn:register m :norm1 (nn:layer-norm dims))
    (nn:register m :attention (nn:multi-head-attention dims heads))
    (nn:register m :norm2 (nn:layer-norm dims))
    (nn:register m :mlp (nn:sequential (nn:linear dims (* 4 dims)) #'nn:gelu (nn:linear (* 4 dims) dims)))
    m))

(defmethod nn:forward ((m gpt-block) &rest args)
  (destructuring-bind (x) args
    (let* ((h (funcall (nn:child m :norm1) x))
           (x (mx:+ x (funcall (nn:child m :attention) h h h :mask :causal))))
      (mx:+ x (funcall (nn:child m :mlp) (funcall (nn:child m :norm2) x))))))

(nn:defmodule gpt () ((context :initarg :context :reader context)))

(defun make-gpt (vocab &key (dims 256) (heads 4) (layers 4) (context 128))
  (let ((m (make-instance 'gpt :context context)))
    (nn:register m :token-embedding (nn:embedding vocab dims))
    (nn:register m :position-embedding (nn:embedding context dims))
    (nn:register m :blocks (loop repeat layers collect (make-block dims heads)))
    (nn:register m :norm (nn:layer-norm dims))
    (nn:register m :head (nn:linear dims vocab))
    m))

(defmethod nn:forward ((m gpt) &rest args)
  (destructuring-bind (tokens) args
    (let ((x (mx:+ (funcall (nn:child m :token-embedding) tokens)
                   (funcall (nn:child m :position-embedding) (mx:arange (mx:dim tokens 1))))))
      (dolist (b (nn:child m :blocks))
        (setf x (funcall b x)))
      (funcall (nn:child m :head) (funcall (nn:child m :norm) x)))))

;;; ------------------------------------------------------------------
;;; Training and sampling

(defun loss (model x y)
  (nn:cross-entropy (funcall model x) y :reduction :mean))

(defun evaluate (model data &key (batches 10) (batch-size 32))
  (nn:train-mode model nil)
  (prog1 (/ (loop repeat batches
                  sum (mx:with-scope ()
                        (multiple-value-bind (x y) (batch data batch-size (context model))
                          (mx:item (loss model x y)))))
            batches)
    (nn:train-mode model t)))

(defun train (model corpus &key (steps 1500) (batch-size 32) (learning-rate 1e-3) (report 250))
  (let* ((warmup 100)
         (opt (optim:adamw (optim:join-schedules
                            (list (optim:linear-schedule 1e-6 learning-rate warmup)
                                  (optim:cosine-decay learning-rate (- steps warmup) :end (/ learning-rate 10)))
                            (list warmup))
                           :weight-decay 0.1))
         (step (nn:value-and-grad model (lambda (x y) (loss model x y))))
         (start (get-internal-real-time)))
    (format t "~&~:D parameters~%" (nn:parameter-count model))
    (dotimes (i steps)
      (mx:with-scope ()
        (multiple-value-bind (x y) (batch (corpus-train corpus) batch-size (context model))
          (multiple-value-bind (l grads) (funcall step x y)
            (optim:update opt model (optim:clip-grad-norm grads 1.0))
            (mx:eval l (nn:parameters model) (optim:state opt))
            (when (or (zerop (mod (1+ i) report)) (zerop i))
              (format t "~&step ~5D  train loss ~,3F  valid loss ~,3F  lr ~,2E  ~,1Fs~%"
                      (1+ i) (mx:item l) (evaluate model (corpus-valid corpus))
                      (optim:learning-rate opt)
                      (/ (- (get-internal-real-time) start) internal-time-units-per-second)))))))
    model))

(defun sample (model corpus &key (prompt "ROMEO:") (length 400) (temperature 0.8))
  "Generate LENGTH characters after PROMPT."
  (nn:train-mode model nil)
  (let ((ids (map 'list (lambda (c) (gethash c (corpus-index corpus))) prompt))
        (chars (corpus-chars corpus)))
    (with-output-to-string (out)
      (write-string prompt out)
      (dotimes (i length)
        (mx:with-scope ()
          (let* ((window (last ids (context model)))
                 (logits (mx:ref (funcall model (mx:from-lisp (list window) :dtype :int32)) 0 -1))
                 (next (mx:item (random:categorical (mx:multiply logits (/ 1.0 temperature))))))
            (setf ids (append ids (list next)))
            (write-char (char chars next) out)))))))

(defun main (&key (steps 1500))
  (random:seed 42)
  (let* ((corpus (make-dataset (load-text)))
         (model (make-gpt (length (corpus-chars corpus)))))
    (format t "~&Tiny Shakespeare: ~:D characters, vocabulary of ~D~%"
            (+ (mx:size (corpus-train corpus)) (mx:size (corpus-valid corpus))) (length (corpus-chars corpus)))
    (train model corpus :steps steps)
    (format t "~&~%--- sample ---~%~A~%" (sample model corpus))))
