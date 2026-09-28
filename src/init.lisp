;;;; init.lisp -- per-process initialization

(in-package :mlx.impl)

(defun mlx:reinitialize ()
  "Reset per-process state: install the error handler and forget cached
streams.  Runs at load time and again when a saved image starts."
  (setf *last-error-message* nil *callback-error* nil)
  (reset-stream-cache)
  (clear-scalar-cache :free nil)
  (install-error-handler)
  t)

(defun clear-foreign-state-before-save ()
  ;; cached handles would be dangling pointers in a new process
  (reset-stream-cache)
  (clear-scalar-cache))

(pushnew 'mlx:reinitialize sb-ext:*init-hooks*)
(pushnew 'clear-foreign-state-before-save sb-ext:*save-hooks*)

(mlx:reinitialize)
