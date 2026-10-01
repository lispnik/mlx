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
    (:loop-bare "+ repair loop, errors and failures only" :attempts 4 :diagnostics nil)
    (:loop "+ repair loop (4 attempts)" :attempts 4)
    (:best-of-4 "+ 4 candidates per attempt" :attempts 4 :candidates 4))
  "(key description write-lisp-arguments...), each adding one technique.
:LOOP-BARE is :LOOP without the call traces in its feedback.")

(defun project-file (name) (asdf:system-relative-pathname "mlx" name))

(defun load-tasks ()
  (with-open-file (in (project-file "bench/lisp-tasks.sexp"))
    (let ((*read-eval* nil)) (read in))))

(defun recorded-results ()
  "Results already in bench/results.sexp, as a list of plists."
  (let ((file (project-file "bench/results.sexp")))
    (when (probe-file file)
      (with-open-file (in file)
        (let ((*read-eval* nil))
          (loop for r = (read in nil in) until (eq r in) collect r))))))

(defun run (&key (model "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit") adapters
                 (configs (remove :loop-bare (mapcar #'first *configs*))) tasks (seed 0) (resume t))
  "Run the benchmark with MODEL, plus ADAPTERS (from LLM:SAVE-ADAPTERS) if
given.  With RESUME, results already recorded for the same model, adapters,
configuration and task are reused, so an interrupted run can be continued."
  (let* ((lm (let ((lm (llm:load-model model))) (if adapters (llm:load-adapters lm adapters) lm)))
         (all (load-tasks))
         (tasks (if tasks (remove-if-not (lambda (tk) (member (getf tk :name) tasks :test #'string=)) all) all))
         (adapter-name (and adapters (namestring adapters)))
         (recorded (and resume (recorded-results)))
         (table '()))
    (dolist (key configs)
      (destructuring-bind (description &rest arguments) (rest (assoc key *configs*))
        (let ((passed 0) (seconds 0) (attempts 0))
          (format t "~&~%== ~A~%" description)
          (dolist (task tasks)
            (mlx.random:seed seed)
            (let ((start (get-internal-real-time))
                  (old (find-if (lambda (r) (and (equal (getf r :model) model)
                                                 (equal (getf r :adapters) adapter-name)
                                                 (eq (getf r :config) key)
                                                 (equal (getf r :task) (getf task :name))))
                                recorded)))
              (multiple-value-bind (code ok result used)
                  (cond (old (values nil (getf old :ok) nil (getf old :attempts)))
                        ;; one task's failure must not end the run
                        (t (handler-case (apply #'llm:write-lisp lm (getf task :task) :tests (getf task :tests)
                                                :stream nil arguments)
                             (error (e)
                               (format t "~&  error in ~A: ~A~%" (getf task :name) e)
                               (values nil nil nil 0)))))
                (declare (ignore code result))
                (let ((time (if old
                                (getf old :seconds)
                                (/ (- (get-internal-real-time) start) internal-time-units-per-second))))
                  (incf seconds time)
                  (when ok (incf passed) (incf attempts used))
                  (format t "~&  ~:[FAIL~;pass~] ~20A ~5,1Fs~:[~*~; (attempt ~D)~]~%"
                          ok (getf task :name) time (and ok (> used 1)) used)
                  (finish-output)
                  (unless old
                    (with-open-file (out (project-file "bench/results.sexp") :direction :output
                                                                            :if-exists :append :if-does-not-exist :create)
                      (with-standard-io-syntax
                        (let ((*print-readably* nil))
                          (print (list :model model :adapters adapter-name
                                       :config key :task (getf task :name) :ok ok
                                       :attempts used :seconds (float time))
                                 out)))))))))
          (push (list description passed (length tasks) seconds) table)
          (format t "~&  ~D/~D passed, ~,1Fs in all~%" passed (length tasks) seconds))))
    (format t "~&~%~A~@[ + ~A~]~%~40A ~8A ~10A~%" model adapters "configuration" "passed" "time")
    (loop for (description passed total seconds) in (reverse table)
          do (format t "~40A ~3D/~2D ~8,1Fs~%" description passed total seconds))
    (reverse table)))
