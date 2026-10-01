;;;; train/lisp-writer-lora.lisp -- distil the 7B Lisp writer into a smaller model
;;;;
;;;;   1. MAKE-DATA: the 7B writer (repair loop, 4 candidates) solves the
;;;;      exercises in train/lisp-train-tasks.sexp; the conversations that
;;;;      ended in code passing the tests go to train/data.sexp.
;;;;   2. TRAIN: LoRA adapters for a smaller model on those conversations
;;;;      (QLoRA on the 4-bit weights), saved under train/adapters/.
;;;;   3. Evaluate with the benchmark, which never saw these exercises:
;;;;        (lisp-writer-bench:run :model BASE :adapters FILE)
;;;;
;;;; Usage, from the project root:
;;;;   sbcl --load train/lisp-writer-lora.lisp --eval '(lisp-writer-lora:make-data)' --quit
;;;;   sbcl --load train/lisp-writer-lora.lisp --eval '(lisp-writer-lora:train)' --quit

(asdf:load-system "mlx/llm")

(defpackage :lisp-writer-lora
  (:use :cl)
  (:local-nicknames (:llm :mlx.llm))
  (:export #:make-data #:train #:*teacher* #:*student*))

(in-package :lisp-writer-lora)

(defparameter *teacher* "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
(defparameter *student* "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit")

(defun project-file (name) (asdf:system-relative-pathname "mlx" name))

(defun read-all (file)
  (with-open-file (in file)
    (let ((*read-eval* nil))
      (loop for form = (read in nil in) until (eq form in) collect form))))

(defun make-data (&key (model *teacher*) (candidates 4) (attempts 4))
  "Solve every training exercise with MODEL; append the successful
conversations to train/data.sexp.  Returns the number solved in all."
  (let ((lm (llm:load-model model))
        (tasks (first (read-all (project-file "train/lisp-train-tasks.sexp"))))
        (solved 0))
    ;; continue an interrupted run: skip exercises already solved
    (let ((done (and (probe-file (project-file "train/data.sexp"))
                     (mapcar (lambda (d) (getf d :task)) (read-all (project-file "train/data.sexp"))))))
      (setf solved (count-if (lambda (task) (member (getf task :name) done :test #'string=)) tasks)
            tasks (remove-if (lambda (task) (member (getf task :name) done :test #'string=)) tasks)))
    (dolist (task tasks)
      (mlx.random:seed 0)
      (multiple-value-bind (code ok result used messages)
          (handler-case (llm:write-lisp lm (getf task :task) :tests (getf task :tests) :stream nil
                                                             :candidates candidates :attempts attempts)
            (error (e) (format t "~&  error: ~A~%" e) (values nil nil nil 0 nil)))
        (declare (ignore result))
        (format t "~&  ~:[FAIL~;ok  ~] ~A~@[ (attempt ~D)~]~%" ok (getf task :name) (and ok (> used 1) used))
        (finish-output)
        (when ok
          (incf solved)
          (with-open-file (out (project-file "train/data.sexp") :direction :output
                                                                :if-exists :append :if-does-not-exist :create)
            (with-standard-io-syntax
              (let ((*print-readably* nil))
                (print (list :task (getf task :name) :messages messages :reply code) out)))))))
    (format t "~&~D exercises solved in all~%" solved)
    solved))

(defun train (&key (model *student*) (iterations 300) (batch-size 4) (learning-rate 1e-4)
                   (rank 8) (num-layers 16) (output "train/adapters/qwen2.5-coder-3b-lisp.safetensors"))
  "Train adapters for MODEL on train/data.sexp and save them to OUTPUT."
  (let* ((lm (llm:load-model model))
         (tokenizer (llm:model-tokenizer lm))
         (examples (mapcar (lambda (d) (llm:chat-example tokenizer (getf d :messages) (getf d :reply)))
                           (read-all (project-file "train/data.sexp"))))
         (file (project-file output)))
    (format t "~&~D training conversations~%" (length examples))
    (llm:lora-train lm examples :iterations iterations :batch-size batch-size :learning-rate learning-rate
                                :rank rank :num-layers num-layers :seed 0)
    (ensure-directories-exist file)
    (llm:save-adapters lm file)
    (format t "~&adapters saved to ~A~%" file)
    file))
