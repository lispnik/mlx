;;;; op.lisp -- DEFINE-OP: the macro behind every generated array operation
;;;;
;;;; A spec such as
;;;;
;;;;   (define-op mlx:sum-axis mlx-ffi:mlx-sum-axis
;;;;     :returns (:array)
;;;;     :args ((a :array) (axis :int) (keepdims :bool :default nil))
;;;;     :stream t)
;;;;
;;;; becomes (defun mlx:sum-axis (a axis &key (keepdims nil) stream) ...),
;;;; which converts each argument according to its kind, calls the C
;;;; function with fresh result slots and the resolved stream, checks the
;;;; status, and wraps the results.
;;;;
;;;; Argument kinds:
;;;;   :array :array-or-null   MLX-ARRAY or Lisp data (NIL -> null for -or-null)
;;;;   :arrays                 list of arrays       -> mlx_vector_array
;;;;   :ints :ints-or-null :int64s   integer list    -> pointer + count
;;;;   :int :size :uint64 :bool :float :double :string
;;;;   :dtype                  dtype keyword
;;;;   :optional-int :optional-float :optional-dtype   value or NIL
;;;;   :fft-norm               :backward :ortho :forward
;;;;   :group-or-null          distributed group handle or NIL
;;;; Return kinds: :array (one array) and :arrays (a list); several return
;;;; kinds yield multiple values.

(in-package :mlx.impl)

(defparameter +weakly-typed-args+
  '("A" "B" "C" "X" "Y" "A-MIN" "A-MAX" "PAD-VALUE" "VALUES" "UPDATE" "UPDATES" "SRC")
  "Arguments whose Lisp scalar values adopt the dtype of the operation's
array arguments (MLX's weak scalar typing), so (add half-array 1.0) stays
float16.  Index-like arguments are deliberately excluded.")

(defun weak-like-dtype (&rest args)
  "Dtype a Lisp scalar among ARGS should adopt, or NIL."
  (when (some (lambda (a) (typep a '(or number (eql t)))) args)
    (let ((arr (find-if (lambda (a) (typep a 'mlx:mlx-array)) args)))
      (and arr (mlx:dtype arr)))))

(defun normalize-int-list (x)
  (etypecase x
    (integer (list x))
    (list x)
    (vector (coerce x 'list))))

(defun call-with-int-buffer (ints type nullable function)
  (if (and nullable (null ints))
      (funcall function (cffi:null-pointer) 0)
      (let* ((list (normalize-int-list ints))
             (n (length list)))
        (cffi:with-foreign-object (buf type (max n 1))
          (loop for x in list for i from 0 do (setf (cffi:mem-aref buf type i) x))
          (funcall function buf n)))))

(defmacro with-int-buffer ((ptr count form &key (type :int) nullable) &body body)
  `(call-with-int-buffer ,form ,type ,nullable
                         (lambda (,ptr ,count)
                           (declare (ignorable ,count))
                           ,@body)))

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun arg-conversion (var kind like)
    "Returns (values foreign-arg-forms wrapper), WRAPPER being a function
from a body form to a form that binds whatever the foreign args need."
    (let ((g (gensym (symbol-name var))) (n (gensym "N")))
      (ecase kind
        ((:array :array-or-null)
         (values (list g) (lambda (body)
                           `(with-array-arg (,g ,var ,like ,(eq kind :array-or-null)) ,body))))
        (:arrays
         (values (list g) (lambda (body) `(with-vector-array (,g ,var) ,body))))
        ((:ints :ints-or-null :int64s)
         (values (list g n)
                 (lambda (body)
                   `(with-int-buffer (,g ,n ,var :type ,(if (eq kind :int64s) :int64 :int)
                                                 :nullable ,(eq kind :ints-or-null))
                      ,body))))
        ((:int :size :uint64 :bool :string :fft-norm) (values (list var) #'identity))
        (:float (values (list `(float ,var 1f0)) #'identity))
        (:double (values (list `(float ,var 1d0)) #'identity))
        (:dtype (values (list `(check-dtype ,var)) #'identity))
        (:optional-int (values (list `(pack-optional-int ,var)) #'identity))
        (:optional-float (values (list `(pack-optional-float ,var)) #'identity))
        (:optional-dtype (values (list `(pack-optional-dtype ,var)) #'identity))
        (:group-or-null (values (list `(if ,var (ptr ,var) (cffi:null-pointer))) #'identity))))))

(defun wrap-result (kind pointer)
  (ecase kind
    (:array (%wrap-mlx-array pointer))
    (:arrays (vector-array->list pointer :free t))))

(defun free-result (kind pointer)
  (unless (cffi:null-pointer-p pointer)
    (ecase kind
      (:array (ffi:mlx-array-free pointer))
      (:arrays (ffi:mlx-vector-array-free pointer)))))

(defmacro call-with-results ((slots returns) call-form operation)
  "Run CALL-FORM (which receives result slot pointers via SLOTS) and wrap
the results according to RETURNS."
  (let ((n (length returns)) (ok (gensym "OK")))
    `(cffi:with-foreign-object (,slots :pointer ,n)
       ,@(loop for i below n collect `(setf (cffi:mem-aref ,slots :pointer ,i) (cffi:null-pointer)))
       (let ((,ok nil))
         (unwind-protect
              (progn
                (check ,call-form ,operation)
                (setf ,ok t)
                (values ,@(loop for kind in returns for i from 0
                                collect `(wrap-result ,kind (cffi:mem-aref ,slots :pointer ,i)))))
           (unless ,ok
             ,@(loop for kind in returns for i from 0
                     collect `(free-result ,kind (cffi:mem-aref ,slots :pointer ,i)))))))))

(defmacro define-op (name c-function &key returns args stream doc)
  (let* ((required (remove-if #'cddr args))
         (keyed (remove-if-not #'cddr args))
         (supplied (mapcar (lambda (a) (gensym (format nil "~A-SUPPLIED" (first a)))) keyed))
         (like (gensym "LIKE"))
         (slots (gensym "SLOTS"))
         (weak (loop for (var kind) in args
                     when (and (member kind '(:array :array-or-null))
                               (member (symbol-name var) +weakly-typed-args+ :test #'string=))
                       collect var))
         (lambda-list
           `(,@(mapcar #'first required)
             &key
             ,@(loop for a in keyed for s in supplied
                     collect (destructuring-bind (var kind &key default computed) a
                               (declare (ignore kind))
                               (if computed `(,var nil ,s) `(,var ,default))))
             ,@(when stream '(stream))))
         ;; right to left, so a computed default may use later arguments
         ;; (fftn's N is computed from its AXES)
         (computed-forms
           (reverse
            (loop for a in keyed for s in supplied
                  for computed = (getf (cddr a) :computed)
                  when computed collect `(unless ,s (setf ,(first a) ,computed)))))
         (foreign-args '())
         (wrappers '()))
    (loop for (var kind) in args
          do (multiple-value-bind (fargs wrapper)
                 (arg-conversion var kind (if (member var weak) like nil))
               (setf foreign-args (append foreign-args fargs))
               (push wrapper wrappers)))
    (let ((body `(call-with-results (,slots ,returns)
                   (,c-function ,@(loop for i below (length returns)
                                        collect `(cffi:mem-aptr ,slots :pointer ,i))
                                ,@foreign-args
                                ,@(when stream '((resolve-stream stream))))
                   ,(string-downcase (symbol-name name)))))
      (dolist (w wrappers) (setf body (funcall w body)))
      `(defun ,name ,lambda-list
         ,@(when doc (list doc))
         ,@computed-forms
         ;; mask once here so the op's several raw calls don't each pay for it
         (ffi:with-float-traps-masked*
           (let ((,like ,(if weak `(weak-like-dtype ,@weak) nil)))
             (declare (ignorable ,like))
             ,body))))))
