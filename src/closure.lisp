;;;; closure.lisp -- Lisp functions as mlx-c closures
;;;;
;;;; mlx-c closures carry a void* payload.  We register the Lisp function in
;;;; a table and pass its integer id as the payload; a static trampoline
;;;; callback looks it up and calls it.  MLX calls the payload destructor
;;;; when it drops the closure, which unregisters the function.
;;;;
;;;; A Lisp error inside a callback must not unwind through C++ frames, so
;;;; trampolines catch it, stash it in *CALLBACK-ERROR* and return failure;
;;;; the enclosing CHECK re-signals the original condition.

(in-package :mlx.impl)

(defvar *callbacks* (make-hash-table))
(defvar *callbacks-lock* (sb-thread:make-mutex :name "mlx callback registry"))
(defvar *next-callback-id* 0)

(defun register-callback (object)
  "Store OBJECT; returns a payload pointer identifying it."
  (sb-thread:with-mutex (*callbacks-lock*)
    (let ((id (incf *next-callback-id*)))
      (setf (gethash id *callbacks*) object)
      (cffi:make-pointer id))))

(defun callback-object (payload)
  (sb-thread:with-mutex (*callbacks-lock*)
    (or (gethash (cffi:pointer-address payload) *callbacks*)
        (error "MLX invoked an unregistered callback (payload ~D)."
               (cffi:pointer-address payload)))))

(defun unregister-callback (payload)
  (sb-thread:with-mutex (*callbacks-lock*)
    (remhash (cffi:pointer-address payload) *callbacks*)))

(cffi:defcallback release-payload :void ((payload :pointer))
  (unregister-callback payload))

(defmacro guarded-callback (&body body)
  "Run BODY for a C callback returning an int status."
  `(handler-case (progn ,@body 0)
     (error (e)
       (setf *callback-error* e)
       1)))

(defun leaves-as-arrays (tree)
  "Flatten a callback's result TREE to a list of arrays."
  (mapcar #'mlx:ensure-array (mlx:tree-flatten tree)))

(defun int-array->list (pointer count)
  (loop for i below count collect (cffi:mem-aref pointer :int i)))

;;; Trampolines, one per closure flavour

(cffi:defcallback closure-trampoline :int
    ((res :pointer) (input ffi:mlx-vector-array) (payload :pointer))
  (guarded-callback
    (let ((fn (callback-object payload)))
      (set-vector-array-out res (leaves-as-arrays (funcall fn (vector-array->list input)))))))

(cffi:defcallback closure-kwargs-trampoline :int
    ((res :pointer) (input ffi:mlx-vector-array) (kwargs ffi:mlx-map-string-to-array)
     (payload :pointer))
  (guarded-callback
    (let ((fn (callback-object payload)))
      (set-vector-array-out res (leaves-as-arrays
                                 (funcall fn (vector-array->list input)
                                          (map-string-to-array->alist kwargs)))))))

(cffi:defcallback closure-custom-trampoline :int
    ((res :pointer) (primals ffi:mlx-vector-array) (cotangents ffi:mlx-vector-array)
     (outputs ffi:mlx-vector-array) (payload :pointer))
  (guarded-callback
    (let ((fn (callback-object payload)))
      (set-vector-array-out res (leaves-as-arrays
                                 (funcall fn (vector-array->list primals)
                                          (vector-array->list cotangents)
                                          (vector-array->list outputs)))))))

(cffi:defcallback closure-custom-jvp-trampoline :int
    ((res :pointer) (primals ffi:mlx-vector-array) (tangents ffi:mlx-vector-array)
     (argnums :pointer) (num :size) (payload :pointer))
  (guarded-callback
    (let ((fn (callback-object payload)))
      (set-vector-array-out res (leaves-as-arrays
                                 (funcall fn (vector-array->list primals)
                                          (vector-array->list tangents)
                                          (int-array->list argnums num)))))))

(cffi:defcallback closure-custom-vmap-trampoline :int
    ((res-arrays :pointer) (res-axes :pointer) (inputs ffi:mlx-vector-array)
     (axes :pointer) (num :size) (payload :pointer))
  (guarded-callback
    (let ((fn (callback-object payload)))
      (multiple-value-bind (outputs out-axes)
          (funcall fn (vector-array->list inputs) (int-array->list axes num))
        (set-vector-array-out res-arrays (leaves-as-arrays outputs))
        (let ((v (list->vector-int out-axes)))
          (unwind-protect (check (ffi:mlx-vector-int-set res-axes v))
            (ffi:mlx-vector-int-free v)))))))

;;; Closure handles

(define-handle-type mlx-closure ffi:mlx-closure-free)
(define-handle-type mlx-closure-kwargs ffi:mlx-closure-kwargs-free)
(define-handle-type mlx-closure-custom ffi:mlx-closure-custom-free)
(define-handle-type mlx-closure-custom-jvp ffi:mlx-closure-custom-jvp-free)
(define-handle-type mlx-closure-custom-vmap ffi:mlx-closure-custom-vmap-free)
(define-handle-type mlx-closure-value-and-grad ffi:mlx-closure-value-and-grad-free)

(defun make-closure (function)
  "An mlx closure calling FUNCTION with a list of input arrays; FUNCTION
returns an array, a number, or a tree of them."
  (%wrap-mlx-closure
   (ffi:mlx-closure-new-func-payload (cffi:callback closure-trampoline)
                                     (register-callback function)
                                     (cffi:callback release-payload))))

(defun make-closure-kwargs (function)
  "FUNCTION receives (inputs kwargs-alist)."
  (%wrap-mlx-closure-kwargs
   (ffi:mlx-closure-kwargs-new-func-payload (cffi:callback closure-kwargs-trampoline)
                                            (register-callback function)
                                            (cffi:callback release-payload))))

(defun make-closure-custom (function)
  "FUNCTION receives (primals cotangents outputs)."
  (%wrap-mlx-closure-custom
   (ffi:mlx-closure-custom-new-func-payload (cffi:callback closure-custom-trampoline)
                                            (register-callback function)
                                            (cffi:callback release-payload))))

(defun make-closure-custom-jvp (function)
  "FUNCTION receives (primals tangents argnums)."
  (%wrap-mlx-closure-custom-jvp
   (ffi:mlx-closure-custom-jvp-new-func-payload (cffi:callback closure-custom-jvp-trampoline)
                                                (register-callback function)
                                                (cffi:callback release-payload))))

(defun make-closure-custom-vmap (function)
  "FUNCTION receives (inputs axes) and returns (values outputs out-axes)."
  (%wrap-mlx-closure-custom-vmap
   (ffi:mlx-closure-custom-vmap-new-func-payload (cffi:callback closure-custom-vmap-trampoline)
                                                 (register-callback function)
                                                 (cffi:callback release-payload))))

(defmacro with-out-slots ((&rest slots) &body body)
  "Bind each of SLOTS to a pointer to a null-initialised handle slot."
  `(cffi:with-foreign-objects ,(mapcar (lambda (s) `(,s :pointer)) slots)
     ,@(mapcar (lambda (s) `(setf (cffi:mem-ref ,s :pointer) (cffi:null-pointer))) slots)
     ,@body))

(defun apply-closure (closure inputs)
  "Apply the mlx closure handle CLOSURE to the list of arrays INPUTS;
returns the list of output arrays."
  (with-vector-array (in inputs)
    (with-out-slots (res)
      (check (ffi:mlx-closure-apply res (ptr closure) in) "closure apply")
      (vector-array->list (cffi:mem-ref res :pointer) :free t))))

;;; Adapting tree-shaped Lisp functions

(defstruct (tree-closure (:constructor %make-tree-closure))
  closure
  (in-structure nil)
  (out-structure nil))

(defun make-tree-closure (function)
  "Wrap FUNCTION (taking trees of arrays as arguments, returning a tree) as
an mlx closure over flat arrays.  The argument skeleton for a call is set
in IN-STRUCTURE before applying; the result skeleton of the most recent
trace is recorded in OUT-STRUCTURE.  Multiple values are preserved: the
skeleton is that of the list of values."
  (let ((tc (%make-tree-closure)))
    (setf (tree-closure-closure tc)
          (make-closure
           (lambda (inputs)
             (let ((result (multiple-value-list
                            (apply function (mlx:tree-unflatten (tree-closure-in-structure tc)
                                                                inputs)))))
               (multiple-value-bind (leaves structure) (mlx:tree-flatten result)
                 (setf (tree-closure-out-structure tc) structure)
                 leaves)))))
    tc))

(defun flatten-args (args)
  "Returns (values arrays structure) for a list of argument trees."
  (multiple-value-bind (leaves structure) (mlx:tree-flatten args)
    (values (mapcar #'mlx:ensure-array leaves) structure)))
