;;; mlx-complete-tests.el --- Tests for mlx-complete  -*- lexical-binding: t -*-

;; Run: emacs --batch -L emacs -l mlx-complete-tests -f ert-run-tests-batch-and-exit
;; The Lisp side is mocked: these check what is sent and how answers are shown.

(require 'ert)
(require 'mlx-complete)

(defmacro mlx-complete-tests--with-lisp (answer &rest body)
  "Run BODY with a fake Lisp that records the request in `sent' and
answers ANSWER."
  (declare (indent 1))
  `(let ((sent nil) (messages '()))
     (cl-letf (((symbol-function 'mlx-complete--eval-async)
                (lambda (form callback) (setq sent form) (funcall callback ,answer)))
               ((symbol-function 'message)
                (lambda (format &rest args) (push (apply #'format-message format args) messages))))
       ,@body)))

(defun mlx-complete-tests--args (form)
  "The arguments of the EMACS-COMPLETE call in the request FORM."
  (cddr (car (last form))))

(ert-deftest mlx-complete-sends-form-prefix-context-and-suffix ()
  (with-temp-buffer
    (lisp-mode)
    (insert "(defun a () 1)\n\n(defun total (items)\n  (reduce ")
    (save-excursion (insert ")\n\n(defun b () 2)"))
    (mlx-complete-tests--with-lisp '("#'+ items" nil 0.3)
      (mlx-complete-at-point)
      (let ((args (mlx-complete-tests--args sent)))
        (should (equal (nth 0 args) "(defun total (items)\n  (reduce "))
        (should (equal (nth 1 args) "(defun a () 1)\n\n"))
        (should (equal (nth 2 args) ")\n\n(defun b () 2)"))
        (should (equal (nth 3 args) "CL-USER"))
        (should (eq (nth 5 args) :form))))))

(ert-deftest mlx-complete-at-top-level-sends-no-form-prefix ()
  (with-temp-buffer
    (lisp-mode)
    (insert "(defun a () 1)\n\n")
    (mlx-complete-tests--with-lisp '("(defun b () 2)" nil 0.3)
      (mlx-complete-at-point t)
      (let ((args (mlx-complete-tests--args sent)))
        (should (equal (nth 0 args) ""))
        (should (eq (nth 5 args) :line))))))

(ert-deftest mlx-complete-shows-then-accepts ()
  (with-temp-buffer
    (lisp-mode)
    (insert "(defun total (items)\n  (reduce ")
    (mlx-complete-tests--with-lisp '("#'+ items))" nil 0.3)
      (mlx-complete-at-point)
      ;; shown, not inserted
      (should (overlayp mlx-complete--overlay))
      (should (equal (buffer-string) "(defun total (items)\n  (reduce "))
      (should (equal (overlay-get mlx-complete--overlay 'mlx-complete-text) "#'+ items))"))
      (mlx-complete-accept)
      (should-not mlx-complete--overlay)
      (should (equal (buffer-string) "(defun total (items)\n  (reduce #'+ items))")))))

(ert-deftest mlx-complete-reports-compiler-complaints ()
  (with-temp-buffer
    (lisp-mode)
    (insert "(defun f (x)\n  ")
    (mlx-complete-tests--with-lisp '("(frob x))" ("undefined function: frob") 0.2)
      (mlx-complete-at-point)
      (should (string-match-p "undefined function: frob" (car messages))))))

(ert-deftest mlx-complete-ignores-stale-answers ()
  (with-temp-buffer
    (lisp-mode)
    (insert "(defun f (x)\n  ")
    (let (pending)
      (cl-letf (((symbol-function 'mlx-complete--eval-async)
                 (lambda (_form callback) (setq pending callback))))
        (mlx-complete-at-point)
        (insert "x")                      ; the buffer changed meanwhile
        (funcall pending '("(1+ x))" nil 0.2))
        (should-not mlx-complete--overlay)))))

(ert-deftest mlx-complete-shows-errors ()
  (with-temp-buffer
    (lisp-mode)
    (insert "(f ")
    (mlx-complete-tests--with-lisp '("" nil 0.1 "model not found")
      (mlx-complete-at-point)
      (should-not mlx-complete--overlay)
      (should (string-match-p "model not found" (car messages))))))

(ert-deftest mlx-complete-request-names-mlx-llm-at-run-time ()
  ;; the request must read in an image that has not loaded mlx/llm yet
  (let ((printed (prin1-to-string (mlx-complete--request-form "(f " "" "" "CL-USER" "a.lisp" :form))))
    (should-not (string-match-p "mlx\\.llm:" printed))
    (should (string-match-p "asdf:load-system" printed))))

(ert-deftest mlx-complete-tab-accepts-an-answer-that-arrives-between-commands ()
  (with-temp-buffer
    (switch-to-buffer (current-buffer))
    (lisp-mode)
    (mlx-complete-mode 1)
    (insert "(defun f (x)\n  ")
    ;; the answer arrives from the Lisp connection, outside any command
    (run-at-time 0 nil (lambda () (mlx-complete--show "(1+ x))" nil 0.1)))
    (sit-for 0.1)
    (should mlx-complete--overlay)
    (execute-kbd-macro (kbd "TAB"))
    (should (equal (buffer-string) "(defun f (x)\n  (1+ x))"))))

(ert-deftest mlx-complete-other-keys-dismiss-and-tab-indents-as-usual ()
  (with-temp-buffer
    (switch-to-buffer (current-buffer))
    (lisp-mode)
    (mlx-complete-mode 1)
    (insert "(defun f (x)\n  ")
    (run-at-time 0 nil (lambda () (mlx-complete--show "(1+ x))" nil 0.1)))
    (sit-for 0.1)
    (execute-kbd-macro (kbd "y"))
    (should-not mlx-complete--overlay)
    (should (equal (buffer-string) "(defun f (x)\n  y"))))

;;; mlx-complete-tests.el ends here
