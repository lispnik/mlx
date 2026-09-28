;;;; errors.lisp -- error reporting and the foreign-call wrapper

(in-package :mlx.impl)

(define-condition mlx:mlx-error (error)
  ((message :initarg :message :initform nil :reader mlx:mlx-error-message)
   (operation :initarg :operation :initform nil :reader mlx-error-operation))
  (:report (lambda (c s)
             (format s "MLX error~@[ in ~A~]: ~A"
                     (mlx-error-operation c)
                     (or (mlx:mlx-error-message c) "unknown error")))))

(defvar *last-error-message* nil
  "Message most recently delivered to the mlx-c error handler.")

(defvar *callback-error* nil
  "A Lisp condition signalled inside a Lisp callback invoked by MLX.  The
callback reports failure to C; the enclosing call re-signals this.")

;; mlx-c's default handler prints the message and exits the process.  Ours
;; just records it; the failing call's non-zero status turns it into a
;; Lisp error on the Lisp side of the boundary.
(cffi:defcallback error-handler :void ((msg :string) (data :pointer))
  (declare (ignore data))
  (setf *last-error-message* msg))

(defun install-error-handler ()
  (ffi:mlx-set-error-handler (cffi:callback error-handler)
                             (cffi:null-pointer) (cffi:null-pointer)))

(defun signal-mlx-error (operation)
  (let ((callback-error *callback-error*)
        (message *last-error-message*))
    (setf *callback-error* nil *last-error-message* nil)
    (if callback-error
        (error callback-error)
        (error 'mlx:mlx-error :message message :operation operation))))

(defmacro without-float-traps (&body body)
  "MLX (Accelerate, Metal) routinely produces inf/NaN; SBCL would trap them.
The MLX-FFI functions already mask traps; this is for other foreign calls."
  `(ffi:with-float-traps-masked* ,@body))

(defmacro check (form &optional operation)
  "Evaluate FORM, a foreign call returning an mlx-c status (MLX-FFI functions
mask float traps themselves).  Signals MLX-ERROR if the status is non-zero."
  (let ((rc (gensym "RC")))
    `(let ((,rc ,form))
       (unless (eql ,rc 0)
         (signal-mlx-error ,(or operation `',(first form))))
       ,rc)))
