;;;; syntax.lisp -- conveniences: n-ary arithmetic and #M array literals

(in-package :mlx.impl)

;;; ------------------------------------------------------------------
;;; Arithmetic and comparison, with CL's argument conventions.  Arguments
;;; are arrays or Lisp data; scalars follow MLX weak typing.

(defun mlx:+ (&rest arrays)
  "Sum of ARRAYS; (+) is 0."
  (if arrays (reduce #'mlx:add arrays) (mlx:scalar 0)))

(defun mlx:* (&rest arrays)
  "Elementwise product of ARRAYS; (*) is 1."
  (if arrays (reduce #'mlx:multiply arrays) (mlx:scalar 1)))

(defun mlx:- (array &rest more)
  "(- a) negates; (- a b c) is a - b - c."
  (if more (reduce #'mlx:subtract more :initial-value array) (mlx:negative array)))

(defun mlx:/ (array &rest more)
  "(/ a) is 1/a; (/ a b c) is a / b / c."
  (if more (reduce #'mlx:divide more :initial-value array) (mlx:reciprocal array)))

(defun mlx:@ (array &rest more)
  "Matrix product of the arguments, left to right."
  (reduce #'mlx:matmul more :initial-value array))

(defun chain (test arrays)
  "Logical AND of TEST over adjacent pairs of ARRAYS (at least two)."
  (when (< (length arrays) 2) (error "Comparison needs at least two arguments."))
  (reduce #'mlx:logical-and
          (loop for (a b) on arrays while b collect (funcall test a b))))

(defun mlx:< (&rest arrays) "True where each argument is less than the next." (chain #'mlx:less arrays))
(defun mlx:> (&rest arrays) (chain #'mlx:greater arrays))
(defun mlx:<= (&rest arrays) (chain #'mlx:less-equal arrays))
(defun mlx:>= (&rest arrays) (chain #'mlx:greater-equal arrays))
(defun mlx:= (&rest arrays) "True where all arguments are equal." (chain #'mlx:equal arrays))

(defun mlx:/= (&rest arrays)
  "True where all arguments are pairwise different (as CL:/=)."
  (when (< (length arrays) 2) (error "Comparison needs at least two arguments."))
  (reduce #'mlx:logical-and
          (loop for (a . rest) on arrays
                nconc (loop for b in rest collect (mlx:not-equal a b)))))

;;; ------------------------------------------------------------------
;;; #M array literals
;;;
;;;   #M(1 2 3)              int32 vector
;;;   #M((1.0 2) (3 4))      2x2 float32
;;;   #M:float16(1 2 3)      explicit dtype
;;;
;;; The literal reads as a form making a fresh array each time it is
;;; evaluated (arrays can be updated in place with (SETF REF)).

(defun read-array-literal (stream subchar arg)
  (declare (ignore subchar arg))
  (let* ((dtype (when (char= (peek-char nil stream t nil t) #\:)
                  (let ((d (read stream t nil t)))
                    (unless (member d (mlx:dtypes))
                      (error "#M: unknown dtype ~S" d))
                    d)))
         (data (read stream t nil t)))
    (unless (or (listp data) (vectorp data) (numberp data))
      (error "#M expects a list, vector or number, not ~S" data))
    `(mlx:from-lisp ',data ,@(when dtype `(:dtype ,dtype)))))

(defun mlx:enable-array-syntax (&optional (readtable *readtable*))
  "Make #M array literals readable in READTABLE (default: the current one).
Note that the standard readtable cannot be modified; use a copy, e.g.
  (setf *readtable* (copy-readtable)) (mx:enable-array-syntax)
Returns READTABLE."
  (set-dispatch-macro-character #\# #\M #'read-array-literal readtable)
  readtable)

(defun mlx:disable-array-syntax (&optional (readtable *readtable*))
  (set-dispatch-macro-character #\# #\M nil readtable)
  readtable)
