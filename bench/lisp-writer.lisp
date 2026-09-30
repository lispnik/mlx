;;;; bench/lisp-writer.lisp -- how much each technique of WRITE-LISP helps
;;;;
;;;; Runs every task of bench/lisp-tasks.sexp under cumulative
;;;; configurations, from a plain model reply up to the full loop, and
;;;; reports the pass rate and time of each.  Usage, from the project root:
;;;;
;;;;   sbcl --load bench/lisp-writer.lisp --eval '(lisp-writer-bench:run)' --quit
;;;;
;;;; (run :model "..." :configs '(...) :tasks '("flatten" ...)) narrows it.
;;;; Results are appended to bench/results.sexp as they come in.

(asdf:load-system "mlx/llm")

(defpackage :lisp-writer-bench
  (:use :cl)
  (:local-nicknames (:llm :mlx.llm))
  (:export #:run #:*configs*))

(in-package :lisp-writer-bench)

(defparameter *configs*
  '((:plain "plain reply, 1 try" :constrained nil :repair-parens nil :attempts 1)
    (:reader "reader-constrained" :repair-parens nil :attempts 1)
    (:parens "+ parens from indentation" :attempts 1)
    (:loop "+ repair loop (4 attempts)" :attempts 4)
    (:best-of-4 "+ 4 candidates per attempt" :attempts 4 :candidates 4))
  "(key description write-lisp-arguments...), each adding one technique.")

(defun project-file (name) (asdf:system-relative-pathname "mlx" name))

(defun load-tasks ()
  (with-open-file (in (project-file "bench/lisp-tasks.sexp"))
    (let ((*read-eval* nil)) (read in))))

(defun run (&key (model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
                 (configs (mapcar #'first *configs*)) tasks (seed 0))
  (let* ((lm (llm:load-model model))
         (all (load-tasks))
         (tasks (if tasks (remove-if-not (lambda (tk) (member (getf tk :name) tasks :test #'string=)) all) all))
         (table '()))
    (dolist (key configs)
      (destructuring-bind (description &rest arguments) (rest (assoc key *configs*))
        (let ((passed 0) (seconds 0) (attempts 0))
          (format t "~&~%== ~A~%" description)
          (dolist (task tasks)
            (mlx.random:seed seed)
            (let ((start (get-internal-real-time)))
              (multiple-value-bind (code ok result used)
                  (apply #'llm:write-lisp lm (getf task :task) :tests (getf task :tests) :stream nil arguments)
                (declare (ignore code result))
                (let ((time (/ (- (get-internal-real-time) start) internal-time-units-per-second)))
                  (incf seconds time)
                  (when ok (incf passed) (incf attempts used))
                  (format t "~&  ~:[FAIL~;pass~] ~20A ~5,1Fs~:[~*~; (attempt ~D)~]~%"
                          ok (getf task :name) time (and ok (> used 1)) used)
                  (finish-output)
                  (with-open-file (out (project-file "bench/results.sexp") :direction :output
                                                                          :if-exists :append :if-does-not-exist :create)
                    (with-standard-io-syntax
                      (let ((*print-readably* nil))
                        (print (list :model model :config key :task (getf task :name) :ok ok
                                     :attempts used :seconds (float time))
                               out))))))))
          (push (list description passed (length tasks) seconds) table)
          (format t "~&  ~D/~D passed, ~,1Fs in all~%" passed (length tasks) seconds))))
    (format t "~&~%~A~%~40A ~8A ~10A~%" model "configuration" "passed" "time")
    (loop for (description passed total seconds) in (reverse table)
          do (format t "~40A ~3D/~2D ~8,1Fs~%" description passed total seconds))
    (reverse table)))
