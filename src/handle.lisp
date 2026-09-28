;;;; handle.lisp -- Lisp wrappers owning mlx-c objects
;;;;
;;;; Every wrapped mlx-c object is a HANDLE whose BOX is a cons
;;;; (pointer . free-function).  A finalizer closes over the box (never the
;;;; handle), so it can free the foreign object after the handle is garbage.
;;;; FREE releases it early; a freed box has a NIL car, which makes later
;;;; use signal an error rather than touch freed memory.
;;;;
;;;; MLX memory lives outside the Lisp heap, so the GC does not feel its
;;;; pressure.  In loops, WITH-SCOPE (or explicit FREE) keeps usage flat.
;;;;
;;;; Handles made inside WITH-SCOPE get no finalizer (registering one costs
;;;; more than a small MLX op): the scope frees them deterministically, and
;;;; only survivors that leave the outermost scope are finalized.

(in-package :mlx.impl)

(defstruct (handle (:constructor nil) (:copier nil))
  (box (error "box required") :type cons)
  (finalized nil))

(defmethod print-object ((h handle) stream)
  (print-unreadable-object (h stream :type t :identity t)
    (when (null (car (handle-box h))) (write-string "freed" stream))))

(defvar *scope* nil
  "When non-NIL, a cons whose car collects handles created in the current
WITH-SCOPE.")

(declaim (inline ptr))
(defun ptr (handle)
  "The foreign pointer of HANDLE; errors if it has been freed."
  (or (car (handle-box handle))
      (error "~S has been freed." handle)))

(defun attach-finalizer (handle)
  (unless (handle-finalized handle)
    (setf (handle-finalized handle) t)
    (let ((box (handle-box handle)))
      (tg:finalize handle
                   (lambda ()
                     (let ((p (car box)))
                       (when p
                         (setf (car box) nil)
                         (funcall (cdr box) p)))))))
  handle)

(defun register-handle (handle)
  (if *scope*
      (push handle (car *scope*))
      (attach-finalizer handle))
  handle)

(defun hand-to-scope (handle scope)
  "Move HANDLE (leaving a scope) into SCOPE, or finalize it if SCOPE is NIL."
  (if scope
      (push handle (car scope))
      (attach-finalizer handle)))

(defun handle-set (handles)
  "A membership predicate for the list HANDLES."
  (if (< (length handles) 16)
      (lambda (h) (member h handles :test #'eq))
      (let ((table (make-hash-table :test 'eq)))
        (dolist (h handles) (setf (gethash h table) t))
        (lambda (h) (gethash h table)))))

(defmacro define-handle-type (name free-function &optional (doc ""))
  "Define a handle struct NAME with constructor %WRAP-NAME (pointer)."
  (let ((wrap (intern (format nil "%WRAP-~A" (symbol-name name))))
        (make (intern (format nil "%MAKE-~A" (symbol-name name)))))
    `(progn
       (defstruct (,name (:include handle) (:constructor ,make (box)) (:copier nil))
         ,doc)
       (defun ,wrap (pointer)
         (register-handle (,make (cons pointer (lambda (p) (,free-function p)))))))))

(defun mlx:free (&rest handles)
  "Release the foreign objects of HANDLES now.  Idempotent.  Lists (and
other trees) of handles are walked.  Returns NIL."
  (labels ((walk (x)
             (typecase x
               (handle (let* ((box (handle-box x)) (p (car box)))
                         (when p
                           (setf (car box) nil)
                           (funcall (cdr box) p))))
               (cons (walk (car x)) (walk (cdr x)))
               ((and vector (not string)) (map nil #'walk x))
               (hash-table (maphash (lambda (k v) (declare (ignore k)) (walk v)) x)))))
    (walk handles)
    nil))

(defun mlx:freed-p (handle)
  (null (car (handle-box handle))))

(defun steal-pointer (handle)
  "Take ownership of HANDLE's pointer, leaving HANDLE freed."
  (prog1 (ptr handle) (setf (car (handle-box handle)) nil)))

(defun mlx:keep (&rest handles)
  "Exempt HANDLES from the innermost WITH-SCOPE (they move to the enclosing
scope, if any).  Returns the first handle."
  (when *scope*
    (let* ((kept (collect-handles handles))
           (keptp (handle-set kept))
           (moved '()))
      (setf (car *scope*) (remove-if (lambda (h) (when (funcall keptp h) (push h moved) t))
                                     (car *scope*)))
      (dolist (h moved) (hand-to-scope h (cdr *scope*)))))
  (first handles))

(defun mlx:persist (&rest handles)
  "Exempt HANDLES (or trees of them) from every enclosing WITH-SCOPE: they
live until freed explicitly or garbage collected.  For arrays stored in
long-lived objects -- model parameters, caches, optimizer state.  Returns
the first handle."
  (let* ((hs (collect-handles handles))
         (in-hs (handle-set hs)))
    (loop for scope = *scope* then (cdr scope)
          while scope
          do (setf (car scope) (delete-if in-hs (car scope))))
    (mapc #'attach-finalizer hs))
  (first handles))

(defun collect-handles (tree)
  (let ((out '()))
    (labels ((walk (x)
               (typecase x
                 (handle (push x out))
                 (cons (walk (car x)) (walk (cdr x)))
                 ((and vector (not string)) (map nil #'walk x))
                 (hash-table (maphash (lambda (k v) (declare (ignore k)) (walk v)) x)))))
      (walk tree))
    out))

(defun call-with-scope (thunk)
  (let* ((outer *scope*)
         ;; car: handles made here; cdr: the enclosing scope (for KEEP)
         (scope (cons '() outer))
         (results '()))
    (unwind-protect
         (let ((*scope* scope))
           (setf results (multiple-value-list (funcall thunk))))
      (let ((survivorp (handle-set (collect-handles results))))
        (dolist (h (car scope))
          (if (funcall survivorp h)
              (hand-to-scope h outer)
              (mlx:free h)))))
    (values-list results)))

(defmacro mlx:with-scope (() &body body)
  "Run BODY; free every MLX object created during it, except those in the
returned values (searched through lists, vectors and hash tables) and those
passed to KEEP.  Survivors are handed to the enclosing scope."
  `(call-with-scope (lambda () ,@body)))
