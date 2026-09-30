;;;; cli/main.lisp -- the mlx-cl command-line driver
;;;;
;;;; Build:  make cli        (or: sbcl --eval '(asdf:make "mlx/cli")')
;;;; Run:    bin/mlx-cl --help

(defpackage :mlx-cli
  (:use :cl)
  (:local-nicknames (:mx :mlx) (:random :mlx.random) (:llm :mlx.llm) (:sr :mlx.symreg))
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
     (stream-error (e)
       (if (eq (stream-error-stream e) sb-sys:*stdout*)
           ;; the reader went away (mlx-cl ... | head): stop quietly,
           ;; without flushing into the closed pipe again
           (sb-ext:exit :code 0 :abort t)
           (progn (format *error-output* "error: ~A~%" e)
                  (uiop:quit 1))))
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
              (*readtable* (mx:enable-array-syntax (copy-readtable)))
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
;;; Language models

(defparameter *default-model* "HuggingFaceTB/SmolLM2-135M-Instruct")

(defun model-options ()
  (list (clingon:make-option :string :long-name "model" :short-name #\m :key :model
                                     :description "Hugging Face repo id or local model directory"
                                     :initial-value *default-model*)
        (clingon:make-option :integer :long-name "max-tokens" :short-name #\n :key :max-tokens
                                      :description "maximum tokens to generate" :initial-value 256)
        (clingon:make-option :string :long-name "temp" :short-name #\t :key :temp
                                     :description "sampling temperature (0 = greedy)" :initial-value "0.7")
        (clingon:make-option :string :long-name "top-p" :key :top-p
                                     :description "nucleus sampling threshold" :initial-value "0.9")
        (clingon:make-option :integer :long-name "seed" :key :seed :description "random seed")
        (clingon:make-option :string :long-name "system" :short-name #\s :key :system
                                     :description "system prompt")
        (clingon:make-option :flag :long-name "verbose" :short-name #\v :key :verbose
                                   :description "report speed and memory")
        (clingon:make-option :flag :long-name "no-think" :key :no-think
                                   :description "ask reasoning models (Qwen3) to answer without thinking")))

(defun number-option (cmd key)
  (let ((v (clingon:getopt cmd key)))
    (let ((n (let ((*read-eval* nil)) (read-from-string v))))
      (unless (realp n) (error "--~(~A~) must be a number, not ~S" key v))
      n)))

(defun load-model-reporting (cmd)
  (let ((source (clingon:getopt cmd :model)))
    (format *error-output* "~&Loading ~A...~%" source)
    (llm:load-model source)))

(defun generate-handler (cmd)
  (let ((prompt (format nil "~{~A~^ ~}" (clingon:command-arguments cmd))))
    (when (zerop (length prompt)) (clingon:print-usage-and-exit cmd *error-output*))
    (with-mlx-errors
      (let* ((temperature (number-option cmd :temp))
             (top-p (number-option cmd :top-p))
             (model (load-model-reporting cmd))
             (samples (clingon:getopt cmd :samples)))
        (if (> samples 1)
            ;; one batch: every sample is generated at once
            (loop for text in (llm:generate-batch model (make-list samples :initial-element prompt)
                                                  :max-tokens (clingon:getopt cmd :max-tokens)
                                                  :temperature temperature :top-p top-p
                                                  :seed (clingon:getopt cmd :seed)
                                                  :system (clingon:getopt cmd :system)
                                                  :chat (not (clingon:getopt cmd :raw))
                                                  :thinking (not (clingon:getopt cmd :no-think))
                                                  :verbose (clingon:getopt cmd :verbose))
                  for i from 1
                  do (format t "~&--- sample ~D~%~A~%" i text))
        (llm:generate model prompt
                      :max-tokens (clingon:getopt cmd :max-tokens)
                      :temperature temperature
                      :top-p top-p
                      :seed (clingon:getopt cmd :seed)
                      :system (clingon:getopt cmd :system)
                      :chat (not (clingon:getopt cmd :raw))
                      :thinking (not (clingon:getopt cmd :no-think))
                      :stream *standard-output*
                      :verbose (clingon:getopt cmd :verbose)))
        (fresh-line)))))

(defun generate-command ()
  (clingon:make-command
   :name "generate"
   :description "generate text with a language model (Llama, Qwen2, Mistral, Phi-3, Gemma 2/3)"
   :usage "[options] PROMPT..."
   :options (append (model-options)
                    (list (clingon:make-option :flag :long-name "raw" :key :raw
                                                     :description "use the prompt as is, without the chat template")
                          (clingon:make-option :integer :long-name "samples" :short-name #\N :key :samples
                                                        :description "generate this many replies at once, in one batch"
                                                        :initial-value 1)))
   :examples '(("Ask a question:" . "mlx-cl generate 'What is the capital of France?'")
               ("Another model, greedy:" . "mlx-cl generate -m mlx-community/gemma-3-1b-it-4bit -t 0 'Write a haiku'")
               ("Four samples, generated together:" . "mlx-cl generate -N 4 -t 0.9 'Name a Lisp dialect'"))
   :handler #'generate-handler))

(defun chat-handler (cmd)
  (with-mlx-errors
    (let* ((temperature (number-option cmd :temp))
           (top-p (number-option cmd :top-p))
           (model (load-model-reporting cmd))
           (system (clingon:getopt cmd :system))
           (history '()))
      (format t "~&Chatting with ~A. Type /reset to start over, /quit or Ctrl-D to leave.~%"
              (clingon:getopt cmd :model))
      (loop
        (format t "~&> ")
        (force-output)
        (let ((line (read-line *standard-input* nil)))
          (cond ((or (null line) (string= line "/quit")) (return))
                ((string= line "/reset") (setf history '()) (format t "(conversation cleared)~%"))
                ((zerop (length (string-trim " " line))))
                (t
                 (setf history (append history (list (cons "user" line))))
                 (let* ((tk (llm:model-tokenizer model))
                        (messages (append (and system (list (cons "system" system))) history))
                        (reply (llm:generate model (llm:apply-chat-template
                                                    tk messages :thinking (not (clingon:getopt cmd :no-think)))
                                             :chat nil
                                             :max-tokens (clingon:getopt cmd :max-tokens)
                                             :temperature temperature
                                             :top-p top-p
                                             :seed (clingon:getopt cmd :seed)
                                             :stream *standard-output*
                                             :verbose (clingon:getopt cmd :verbose))))
                   (fresh-line)
                   (setf history (append history (list (cons "assistant" reply))))))))))))

(defun chat-command ()
  (clingon:make-command
   :name "chat"
   :description "chat interactively with a language model"
   :options (model-options)
   :handler #'chat-handler))

(defun download-handler (cmd)
  (let ((repos (clingon:command-arguments cmd)))
    (when (null repos) (clingon:print-usage-and-exit cmd *error-output*))
    (with-mlx-errors
      (dolist (repo repos)
        (format t "~A~%" (uiop:native-namestring (llm:download-model repo)))))))

(defun download-command ()
  (clingon:make-command
   :name "download"
   :description "download Hugging Face models into the local cache (~/.cache/mlx-cl/models)"
   :usage "REPO..."
   :handler #'download-handler))

;;; ------------------------------------------------------------------
;;; lisp: a model writes, runs and repairs Common Lisp

(defparameter *default-lisp-model* "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

(defun lisp-handler (cmd)
  (let ((task (format nil "~{~A~^ ~}" (clingon:command-arguments cmd))))
    (when (zerop (length task)) (clingon:print-usage-and-exit cmd *error-output*))
    (with-mlx-errors
      (let ((model (progn (format *error-output* "~&Loading ~A...~%" (clingon:getopt cmd :model))
                          (llm:load-model (clingon:getopt cmd :model)))))
        (multiple-value-bind (code ok result attempts)
            (llm:write-lisp model task
                            :tests (clingon:getopt cmd :tests)
                            :attempts (clingon:getopt cmd :attempts)
                            :candidates (clingon:getopt cmd :candidates)
                            :timeout (clingon:getopt cmd :timeout)
                            :isolation (if (clingon:getopt cmd :in-process) :in-process :process)
                            :stream (and (clingon:getopt cmd :verbose) *standard-output*))
          (format t "~&~A~%" code)
          (format *error-output* "~&;; ~:[failed after ~D attempt~:P~;works (attempt ~D)~]~@[: ~A~]~%"
                  ok attempts (and (not ok) (or (getf result :error) (first (getf result :failures)))))
          (unless ok (uiop:quit 1)))))))

(defun lisp-command ()
  (clingon:make-command
   :name "lisp"
   :description "have a model write Common Lisp, run it against tests, and repair it"
   :usage "[options] TASK..."
   :options (list (clingon:make-option :string :long-name "model" :short-name #\m :key :model
                                               :description "model (a coder model works best)"
                                               :initial-value *default-lisp-model*)
                  (clingon:make-option :list :long-name "test" :short-name #\T :key :tests
                                             :description "a form that must return true (repeatable)")
                  (clingon:make-option :integer :long-name "attempts" :short-name #\a :key :attempts
                                                :description "tries before giving up" :initial-value 4)
                  (clingon:make-option :integer :long-name "candidates" :short-name #\n :key :candidates
                                                :description "forms generated at once per attempt (one batch); the first that passes wins"
                                                :initial-value 1)
                  (clingon:make-option :integer :long-name "timeout" :key :timeout
                                                :description "seconds allowed per evaluation" :initial-value 10)
                  (clingon:make-option :flag :long-name "in-process" :key :in-process
                                             :description "evaluate in this process instead of a child SBCL")
                  (clingon:make-option :flag :long-name "verbose" :short-name #\v :key :verbose
                                             :description "show every attempt and its result"))
   :examples '(("Write and test a function:" .
                "mlx-cl lisp -v 'Define (flatten tree) returning the atoms of a nested list in order' -T '(equal (flatten (quote (1 (2 (3)) 4))) (quote (1 2 3 4)))'"))
   :handler #'lisp-handler))

;;; ------------------------------------------------------------------
;;; symreg

(defun read-formula (text)
  (let ((*read-eval* nil) (*package* (find-package :mlx.symreg))
        (*read-default-float-format* 'single-float))
    (read-from-string text)))

(defun formula-variables (formula)
  "X0 .. Xk for the highest Xk in FORMULA."
  (let ((top -1))
    (labels ((walk (e)
               (cond ((consp e) (mapc #'walk (rest e)))
                     ((symbolp e)
                      (let ((name (symbol-name e)))
                        (when (and (> (length name) 1) (char-equal (char name 0) #\X)
                                   (every #'digit-char-p (subseq name 1)))
                          (setf top (max top (parse-integer name :start 1)))))))))
      (walk formula))
    (loop for i to top collect (intern (format nil "X~D" i) :mlx.symreg))))

(defun formula-data (cmd)
  "Samples of the --formula: (values rows targets variables)."
  (let* ((formula (read-formula (clingon:getopt cmd :formula)))
         (variables (formula-variables formula))
         (f (sr:expression-function formula variables))
         (low (number-option cmd :low)) (high (number-option cmd :high))
         (noise (number-option cmd :noise))
         (*random-state* (sb-ext:seed-random-state (or (clingon:getopt cmd :seed) 0)))
         (rows (loop repeat (clingon:getopt cmd :samples)
                     collect (loop repeat (max 1 (length variables))
                                   collect (+ low (random (float (- high low) 1f0))))))
         (ys (mapcar (lambda (r) (apply f (subseq r 0 (length variables)))) rows)))
    (when (plusp noise)
      (let* ((mean (/ (reduce #'+ ys) (length ys)))
             (sd (sqrt (/ (reduce #'+ ys :key (lambda (y) (expt (- y mean) 2))) (length ys)))))
        (setf ys (mapcar (lambda (y)            ; Box-Muller
                           (+ y (* noise sd (sqrt (* -2 (log (- 1 (random 1d0))))) (cos (* 2 pi (random 1d0))))))
                         ys))))
    (values rows ys (or variables (list (intern "X0" :mlx.symreg))))))

(defun csv-data (path)
  "Numeric columns of the CSV file at PATH, the last being the target:
(values rows targets variables).  A header row names the variables."
  (let* ((lines (remove-if (lambda (l) (zerop (length (string-trim " " l))))
                           (uiop:read-file-lines path)))
         (split (lambda (l) (mapcar (lambda (f) (string-trim " \"" f))
                                    (uiop:split-string (string-right-trim '(#\Return) l) :separator ","))))
         (first-row (funcall split (first lines)))
         (header (notevery (lambda (f) (realp (ignore-errors (read-formula f)))) first-row))
         (rows (loop for l in (if header (rest lines) lines)
                     collect (mapcar (lambda (f)
                                       (let ((n (ignore-errors (read-formula f))))
                                         (unless (realp n) (error "Not a number in ~A: ~S" path f))
                                         n))
                                     (funcall split l))))
         (nvars (1- (length first-row))))
    (values (mapcar #'butlast rows)
            (mapcar (lambda (r) (car (last r))) rows)
            (if header
                (loop for name in (butlast first-row)
                      collect (intern (string-upcase (substitute #\- #\Space name)) :mlx.symreg))
                (loop for i below nvars collect (intern (format nil "X~D" i) :mlx.symreg))))))

(defun symreg-handler (cmd)
  (let ((formula (clingon:getopt cmd :formula))
        (csv (first (clingon:command-arguments cmd))))
    (unless (or formula csv) (clingon:print-usage-and-exit cmd *error-output*))
    (with-mlx-errors
      (multiple-value-bind (rows ys variables) (if csv (csv-data csv) (formula-data cmd))
        (format *error-output* "~&~:D samples of ~{~(~A~)~^, ~}~%" (length rows) variables)
        (multiple-value-bind (best front)
            (sr:symbolic-regression rows ys
                                    :variables variables
                                    :operators (mapcar #'read-formula
                                                       (uiop:split-string (clingon:getopt cmd :operators)
                                                                          :separator ", "))
                                    :population (clingon:getopt cmd :population)
                                    :generations (clingon:getopt cmd :generations)
                                    :time-limit (clingon:getopt cmd :time-limit)
                                    :max-size (clingon:getopt cmd :max-size)
                                    :scaling (not (clingon:getopt cmd :no-scaling))
                                    :seed (clingon:getopt cmd :seed)
                                    :stream (and (not (clingon:getopt cmd :quiet)) *error-output*))
          (let ((*print-case* :downcase) (*print-pretty* nil) (*package* (find-package :mlx.symreg)))
            (format t "~&~%size  loss (MSE / variance)  expression~%")
            (dolist (c front)
              (format t "~4D  ~21,3,,,,,'EG  ~S~%" (sr:candidate-size c) (sr:candidate-loss c)
                      (mlx.symreg::round-constants (sr:candidate-expression c))))
            (format t "~%best: (lambda ~S ~S)~%" variables (sr:candidate-expression best))))))))

(defun symreg-command ()
  (clingon:make-command
   :name "symreg"
   :description "find a formula fitting data, by genetic programming on the GPU"
   :usage "[options] [DATA.csv]"
   :options (list (clingon:make-option :string :long-name "formula" :short-name #\f :key :formula
                                               :description "generate data from this formula in x0, x1, ... instead of reading a CSV (last column = target)")
                  (clingon:make-option :integer :long-name "samples" :short-name #\n :key :samples
                                                :description "samples to generate" :initial-value 300)
                  (clingon:make-option :string :long-name "low" :key :low
                                               :description "smallest generated input" :initial-value "-3")
                  (clingon:make-option :string :long-name "high" :key :high
                                               :description "largest generated input" :initial-value "3")
                  (clingon:make-option :string :long-name "noise" :key :noise
                                               :description "Gaussian noise added, relative to the target's spread"
                                               :initial-value "0")
                  (clingon:make-option :string :long-name "operators" :short-name #\o :key :operators
                                               :description (format nil "operators to use, from: ~{~(~A~)~^ ~}"
                                                                    (sr:operator-names))
                                               :initial-value (format nil "~{~(~A~)~^ ~}" sr:*default-operators*))
                  (clingon:make-option :integer :long-name "population" :short-name #\p :key :population
                                                :description "expressions per generation" :initial-value 1000)
                  (clingon:make-option :integer :long-name "generations" :short-name #\g :key :generations
                                                :description "maximum generations" :initial-value 200)
                  (clingon:make-option :integer :long-name "time-limit" :short-name #\t :key :time-limit
                                                :description "stop after this many seconds")
                  (clingon:make-option :integer :long-name "max-size" :key :max-size
                                                :description "largest expression, in nodes" :initial-value 30)
                  (clingon:make-option :flag :long-name "no-scaling" :key :no-scaling
                                             :description "judge f itself, not the best a + b f")
                  (clingon:make-option :integer :long-name "seed" :key :seed :description "random seed")
                  (clingon:make-option :flag :long-name "quiet" :short-name #\q :key :quiet
                                             :description "don't report each generation"))
   :examples '(("Rediscover a formula from 300 samples of it:" .
                "mlx-cl symreg -f '(+ (square x0) (* 2.5 (sin x1)))'")
               ("Fit data (the last column is the target):" .
                "mlx-cl symreg -t 60 data.csv"))
   :handler #'symreg-handler))

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
                       (inspect-command) (train-command)
                       (generate-command) (chat-command) (download-command) (lisp-command)
                       (symreg-command))))

(defun main ()
  (clingon:run (top-level-command)))
