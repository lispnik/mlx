;;;; array.lisp -- the MLX-ARRAY type, conversion to and from Lisp data

(in-package :mlx.impl)

(define-handle-type mlx:mlx-array ffi:mlx-array-free
  "An MLX n-dimensional array.  Arrays are lazy: operations build a graph that
is computed on EVAL, or implicitly when data is read (TO-LISP, ITEM, printing).")

;;; ------------------------------------------------------------------
;;; Properties

(defun mlx:ndim (a) (ffi:mlx-array-ndim (ptr a)))
(defun mlx:size (a) "Number of elements." (ffi:mlx-array-size (ptr a)))
(defun mlx:itemsize (a) (ffi:mlx-array-itemsize (ptr a)))
(defun mlx:nbytes (a) (ffi:mlx-array-nbytes (ptr a)))
(defun mlx:dtype (a) (ffi:mlx-array-dtype (ptr a)))

(defun mlx:shape (a)
  "The shape of A as a list of integers."
  (let* ((p (ptr a))
         (n (ffi:mlx-array-ndim p))
         (dims (ffi:mlx-array-shape p)))
    (loop for i below n collect (cffi:mem-aref dims :int i))))

(defun mlx:strides (a)
  "Element strides of A's memory layout (valid once evaluated)."
  (let* ((p (ptr a))
         (n (ffi:mlx-array-ndim p))
         (s (ffi:mlx-array-strides p)))
    (loop for i below n collect (cffi:mem-aref s :size i))))

(defun mlx:dim (a axis)
  "Size of axis AXIS (negative counts from the end)."
  (let ((n (mlx:ndim a)))
    (ffi:mlx-array-dim (ptr a) (if (minusp axis) (+ axis n) axis))))

;;; helpers used by generated default forms

(defun dtype-of (x)
  "Dtype of X: an array's dtype, or the default dtype of a Lisp scalar."
  (if (typep x 'mlx:mlx-array) (mlx:dtype x) (scalar-dtype x)))

(defun shape-of (x)
  (if (typep x 'mlx:mlx-array) (mlx:shape x) '()))

(defun dim (a axis) (mlx:dim a axis))

(defun all-axes (a) (loop for i below (mlx:ndim a) collect i))

(defun axes-dims (a axes &key inverse-real)
  "Sizes of A along AXES.  With INVERSE-REAL, the last is 2*(n-1), the
output length of an inverse real FFT."
  (let ((dims (mapcar (lambda (ax) (mlx:dim a ax)) axes)))
    (if (and inverse-real dims)
        (append (butlast dims) (list (* 2 (1- (car (last dims))))))
        dims)))

;;; ------------------------------------------------------------------
;;; Creating arrays from Lisp data

(defun subsequencep (x)
  "True for a nested sequence inside Lisp data.  NIL is boolean false there."
  (and x (typep x 'sequence) (not (stringp x))))

(defun nested-shape (data)
  "Shape of the nested lists/vectors DATA, read along first elements.
COLLECT-LEAVES checks that the rest agrees."
  (let ((shape '()) (x data))
    (loop while (subsequencep x)
          do (let ((n (length x)))
               (push n shape)
               (when (zerop n) (return))
               (setf x (elt x 0))))
    (nreverse shape)))

(defun collect-leaves (data shape)
  "The leaves of DATA in row-major order as a simple-vector, signalling an
error unless DATA is rectangular with SHAPE."
  (let ((out (make-array (reduce #'* shape)))
        (i 0))
    (declare (type simple-vector out) (type fixnum i))
    (labels ((ragged () (error "Ragged nested sequence: ~S" data))
             (walk (x dims)
               (cond ((null dims)
                      (when (subsequencep x) (ragged))
                      (setf (svref out i) x)
                      (incf i))
                     ((not (or (subsequencep x) (and (null x) (zerop (first dims))))) (ragged))
                     ((listp x)
                      (let ((count 0))
                        (declare (type fixnum count))
                        (dolist (e x) (walk e (rest dims)) (incf count))
                        (unless (= count (first dims)) (ragged))))
                     (t (unless (= (length x) (first dims)) (ragged))
                        (loop for e across x do (walk e (rest dims)))))))
      (walk data shape))
    out))

(defun array-row-major-elements (array)
  "ARRAY's elements as a simple-vector (shared, not copied, when ARRAY is a
simple T array)."
  (if (and (typep array '(simple-array t *)))
      (sb-ext:array-storage-vector array)
      (let ((v (make-array (array-total-size array))))
        (dotimes (i (length v) v)
          (setf (svref v i) (row-major-aref array i))))))

(defun convert-element (x dtype)
  (ecase (dtype-kind dtype)
    (:bool (not (or (null x) (and (numberp x) (zerop x)))))
    ((:signed :unsigned)
     (etypecase x
       (integer x)
       ((member t nil) (if x 1 0))
       (real (error "Cannot store non-integer ~S in an ~(~A~) array." x dtype))))
    (:float (etypecase x
              (real (if (eq dtype :float64) (float x 1d0) (float x 1f0)))
              ((member t nil) (if x 1f0 0f0))))
    (:complex (etypecase x
                (number (coerce x '(complex single-float)))))))

(defun new-data-pointer (buffer shape dtype)
  (let ((rank (length shape)))
    (cffi:with-foreign-object (dims :int (max rank 1))
      (loop for d in shape for i from 0 do (setf (cffi:mem-aref dims :int i) d))
      (without-float-traps
        (ffi:mlx-array-new-data buffer dims rank dtype)))))

(defun lisp-buffer-type (dtype)
  "Element type of a Lisp vector whose memory layout matches DTYPE."
  (ecase dtype
    (:bool '(unsigned-byte 8))            ; C bool is one byte
    (:uint8 '(unsigned-byte 8)) (:uint16 '(unsigned-byte 16))
    (:uint32 '(unsigned-byte 32)) (:uint64 '(unsigned-byte 64))
    (:int8 '(signed-byte 8)) (:int16 '(signed-byte 16))
    (:int32 '(signed-byte 32)) (:int64 '(signed-byte 64))
    (:float32 'single-float) (:float64 'double-float)
    (:complex64 '(complex single-float))))

(declaim (inline to-single to-double to-integer to-bool-byte))
(defun to-single (x)
  (typecase x
    (single-float x)
    (real (float x 1f0))
    (t (if (eq x t) 1f0 (if (null x) 0f0 (error "Cannot store ~S in a float array." x))))))
(defun to-double (x)
  (typecase x
    (double-float x)
    (real (float x 1d0))
    (t (if (eq x t) 1d0 (if (null x) 0d0 (error "Cannot store ~S in a float array." x))))))
(defun to-integer (x)
  (typecase x
    (integer x)
    (t (cond ((eq x t) 1) ((null x) 0)
             (t (error "Cannot store non-integer ~S in an integer array." x))))))
(defun to-bool-byte (x)
  (if (or (null x) (and (numberp x) (zerop x))) 0 1))

(defun staged-buffer (elements dtype)
  "A specialized Lisp vector holding ELEMENTS (a simple-vector) converted
to DTYPE, laid out as MLX expects.  One tight, type-declared loop per dtype."
  (declare (type simple-vector elements) (optimize speed))
  (let ((n (length elements)))
    (macrolet ((fill-as (element-type convert)
                 `(let ((buffer (make-array (max n 1) :element-type ',element-type)))
                    (dotimes (i n buffer)
                      (setf (aref buffer i) (,convert (svref elements i)))))))
      (ecase dtype
        (:float32 (fill-as single-float to-single))
        (:float64 (fill-as double-float to-double))
        (:int32 (fill-as (signed-byte 32) to-integer))
        (:int64 (fill-as (signed-byte 64) to-integer))
        (:int16 (fill-as (signed-byte 16) to-integer))
        (:int8 (fill-as (signed-byte 8) to-integer))
        (:uint8 (fill-as (unsigned-byte 8) to-integer))
        (:uint16 (fill-as (unsigned-byte 16) to-integer))
        (:uint32 (fill-as (unsigned-byte 32) to-integer))
        (:uint64 (fill-as (unsigned-byte 64) to-integer))
        (:bool (fill-as (unsigned-byte 8) to-bool-byte))  ; C bool is one byte
        (:complex64 (let ((buffer (make-array (max n 1) :element-type '(complex single-float))))
                      (dotimes (i n buffer)
                        (setf (aref buffer i) (coerce (svref elements i) '(complex single-float))))))))))

(defun elements->pointer (elements shape dtype)
  "Raw mlx_array pointer holding ELEMENTS (a simple-vector, row-major) as
DTYPE.  The data is staged in a specialized Lisp vector passed by address."
  (case dtype
    ((:float16 :bfloat16)
     (let ((f32 (elements->pointer elements shape :float32)))
       (unwind-protect (steal-pointer (mlx:astype (%wrap-mlx-array-unregistered f32) dtype))
         (ffi:mlx-array-free f32))))
    (t
     (let ((buffer (staged-buffer (coerce elements 'simple-vector) (check-dtype dtype))))
       (cffi:with-pointer-to-vector-data (buf buffer)
         (new-data-pointer buf shape dtype))))))

(defun scalar->pointer (x dtype)
  "Raw mlx_array pointer for the Lisp scalar X as DTYPE, using mlx-c's
scalar constructors when one matches."
  (without-float-traps
    (case dtype
      (:int32 (if (typep x '(signed-byte 32))
                  (ffi:mlx-array-new-int x)
                  (elements->pointer (vector x) '() dtype)))
      (:float32 (if (realp x)
                    (ffi:mlx-array-new-float32 (float x 1f0))
                    (elements->pointer (vector x) '() dtype)))
      (:float64 (if (realp x)
                    (ffi:mlx-array-new-float64 (float x 1d0))
                    (elements->pointer (vector x) '() dtype)))
      (:bool (ffi:mlx-array-new-bool (convert-element x :bool)))
      (:complex64 (let ((c (convert-element x :complex64)))
                    (ffi:mlx-array-new-complex (realpart c) (imagpart c))))
      (t (elements->pointer (vector x) '() dtype)))))

(defun %wrap-mlx-array-unregistered (pointer)
  "A handle without a finalizer, for short-lived internal use; the caller
must free POINTER itself."
  (%make-mlx-array (cons pointer (lambda (p) (declare (ignore p))))))

(defun fast-copy-p (array dtype)
  (and (typep array '(simple-array * *))
       (let ((native (lisp-type-dtype (array-element-type array))))
         (and native (eq native dtype) (not (eq dtype :bool))))))

(defun lisp-array->pointer (array dtype)
  (let* ((dtype (or dtype
                    (lisp-type-dtype (array-element-type array))
                    (infer-dtype (array-row-major-elements array))))
         (shape (array-dimensions array)))
    (check-dtype dtype)
    (if (fast-copy-p array dtype)
        (let ((storage (sb-ext:array-storage-vector array)))
          (cffi:with-pointer-to-vector-data (buf storage)
            (new-data-pointer buf shape dtype)))
        (elements->pointer (array-row-major-elements array) shape dtype))))

(defun nested->pointer (data dtype)
  (let* ((shape (nested-shape data))
         (leaves (collect-leaves data shape)))
    (elements->pointer leaves shape (or dtype (infer-dtype leaves)))))

(defun %from-lisp (data dtype)
  "Raw mlx_array pointer for Lisp DATA (the caller owns it)."
  (etypecase data
    (mlx:mlx-array (let ((p (ptr data)))
                     (if (and dtype (not (eq dtype (mlx:dtype data))))
                         (steal-pointer (mlx:astype data dtype))
                         ;; a new reference to the same array
                         (cffi:with-foreign-object (slot :pointer)
                           (setf (cffi:mem-ref slot :pointer) (ffi:mlx-array-new))
                           (check (ffi:mlx-array-set slot p))
                           (cffi:mem-ref slot :pointer)))))
    (null (elements->pointer #() '(0) (or dtype :float32)))
    (list (nested->pointer data dtype))
    ((or number (eql t))
     (scalar->pointer data (or dtype (scalar-dtype data))))
    (string (error "Cannot make an MLX array from the string ~S." data))
    (vector (if (and (eq (array-element-type data) t) (some #'subsequencep data))
                (nested->pointer data dtype)
                (lisp-array->pointer data dtype)))
    (array (lisp-array->pointer data dtype))))

(defun mlx:from-lisp (data &key dtype)
  "Make an MLX array from Lisp DATA: a number (or T for boolean true), a
Lisp array of any rank, or nested lists/vectors.  DTYPE defaults from the
array's element type or the data: integers -> :int32, reals -> :float32,
complexes -> :complex64, T/NIL -> :bool.  The data is copied.
An MLX-ARRAY is returned as is (or cast, if DTYPE differs)."
  (if (and (typep data 'mlx:mlx-array) (or (null dtype) (eq dtype (mlx:dtype data))))
      data
      (%wrap-mlx-array (%from-lisp data dtype))))

(defun mlx:scalar (x &key dtype)
  "A 0-dimensional array holding the number X."
  (mlx:from-lisp x :dtype dtype))

(defun mlx:ensure-array (x &key dtype)
  "X if it is an MLX-ARRAY (cast to DTYPE if given), else (FROM-LISP X)."
  (mlx:from-lisp x :dtype dtype))

;;; Lisp scalars passed to operations (learning rates, 0, 1, 2...) recur
;;; constantly.  Each would otherwise cost a fresh Metal buffer, so they are
;;; cached by (value . dtype).  MLX arrays are immutable, and the cache's own
;;; reference also stops MLX from donating the buffer to an op's output.

(defvar *scalar-cache* (make-hash-table :test 'equal))
(defvar *scalar-cache-lock* (sb-thread:make-mutex :name "mlx scalar cache"))
(defparameter *scalar-cache-limit* 1024)

(defun clear-scalar-cache (&key (free t))
  "Drop cached scalar arrays (FREE NIL when their pointers are stale, e.g.
in a restarted image)."
  (sb-thread:with-mutex (*scalar-cache-lock*)
    (when free
      (maphash (lambda (k p) (declare (ignore k)) (ffi:mlx-array-free p)) *scalar-cache*))
    (clrhash *scalar-cache*)))

(defun cached-scalar-pointer (x dtype)
  "A shared, cache-owned array pointer for the scalar X as DTYPE."
  (let ((key (cons x dtype)))
    (sb-thread:with-mutex (*scalar-cache-lock*)
      (or (gethash key *scalar-cache*)
          (progn
            (when (>= (hash-table-count *scalar-cache*) *scalar-cache-limit*)
              (maphash (lambda (k p) (declare (ignore k)) (ffi:mlx-array-free p)) *scalar-cache*)
              (clrhash *scalar-cache*))
            (setf (gethash key *scalar-cache*) (scalar->pointer x dtype)))))))

(defun array-arg-pointer (x like &optional nullable)
  "Foreign pointer for an array argument X.  Returns (values pointer temp)
where TEMP is a pointer the caller must free (or NIL).  LIKE is the dtype
that a Lisp scalar should adopt (MLX weak typing), or NIL.  NIL is
accepted (as a null array) only when NULLABLE."
  (typecase x
    (mlx:mlx-array (values (ptr x) nil))
    (null (if nullable
              (values (cffi:null-pointer) nil)
              (error "Expected an array, got NIL.")))
    ((or number (eql t))
     ;; cache-owned: not a temporary for the caller to free
     (values (cached-scalar-pointer x (if like (weak-scalar-dtype x like) (scalar-dtype x)))
             nil))
    (t (let ((p (%from-lisp x nil))) (values p p)))))

(defmacro with-array-arg ((var form &optional like nullable) &body body)
  "Bind VAR to the foreign pointer for the array argument FORM, converting
Lisp data to a temporary array that is freed after BODY."
  (let ((temp (gensym "TEMP")))
    `(multiple-value-bind (,var ,temp) (array-arg-pointer ,form ,like ,nullable)
       (unwind-protect (progn ,@body)
         (when ,temp (ffi:mlx-array-free ,temp))))))

;;; ------------------------------------------------------------------
;;; Reading arrays back into Lisp

(defun eval-pointer (p)
  (check (ffi:mlx-array-eval p) "eval"))

(defun row-contiguous-p (a)
  (let ((expected 1) (ok t))
    (loop for d in (reverse (mlx:shape a))
          for s in (reverse (mlx:strides a))
          do (when (and (> d 1) (/= s expected)) (setf ok nil))
             (setf expected (* expected d)))
    ok))

(defun data-pointer (p dtype)
  (ecase dtype
    (:bool (ffi:mlx-array-data-bool p))
    (:uint8 (ffi:mlx-array-data-uint8 p))
    (:uint16 (ffi:mlx-array-data-uint16 p))
    (:uint32 (ffi:mlx-array-data-uint32 p))
    (:uint64 (ffi:mlx-array-data-uint64 p))
    (:int8 (ffi:mlx-array-data-int8 p))
    (:int16 (ffi:mlx-array-data-int16 p))
    (:int32 (ffi:mlx-array-data-int32 p))
    (:int64 (ffi:mlx-array-data-int64 p))
    (:float32 (ffi:mlx-array-data-float32 p))
    (:float64 (ffi:mlx-array-data-float64 p))
    (:complex64 (ffi:mlx-array-data-complex64 p))))

(defun readable-array (a)
  "An evaluated, row-contiguous array with a Lisp-representable dtype
holding A's values.  Second value: T if it is a new temporary."
  (let ((dtype (mlx:dtype a)))
    (cond ((member dtype '(:float16 :bfloat16))
           (values (let ((b (mlx:astype a :float32))) (eval-pointer (ptr b)) b) t))
          (t (eval-pointer (ptr a))
             (if (row-contiguous-p a)
                 (values a nil)
                 (let ((b (mlx:contiguous a))) (eval-pointer (ptr b)) (values b t)))))))

(defun copy-to-lisp (a)
  "A fresh Lisp array with A's shape and values."
  (multiple-value-bind (b temp) (readable-array a)
    (unwind-protect
         (let* ((dtype (mlx:dtype b))
                (shape (mlx:shape b))
                (n (mlx:size b))
                (src (data-pointer (ptr b) dtype))
                (out (make-array shape :element-type (dtype-lisp-type dtype))))
           (if (eq dtype :bool)
               (dotimes (i n) (setf (row-major-aref out i) (cffi:mem-aref src :bool i)))
               (let ((storage (sb-ext:array-storage-vector out)))
                 (cffi:with-pointer-to-vector-data (dst storage)
                   (cffi:foreign-funcall "memcpy" :pointer dst :pointer src :size (mlx:nbytes b)
                                                  :pointer))))
           out)
      (when temp (mlx:free b)))))

(defun lisp-array->nested-list (array)
  (let ((dims (array-dimensions array)))
    (labels ((build (dims offset)
               (if (null dims)
                   (row-major-aref array offset)
                   (let ((stride (reduce #'* (rest dims))))
                     (loop for i below (first dims)
                           collect (build (rest dims) (+ offset (* i stride))))))))
      (build dims 0))))

(defun mlx:to-lisp (a &key (as :array))
  "Evaluate A and copy its values into Lisp.  AS is :ARRAY (a Lisp array of
A's shape, specialized when possible) or :LIST (nested lists).  A
0-dimensional array yields a plain number.  Booleans become T/NIL."
  (let ((out (copy-to-lisp (mlx:ensure-array a))))
    (cond ((zerop (array-rank out)) (aref out))
          ((eq as :list) (lisp-array->nested-list out))
          (t out))))

(defun mlx:item (a)
  "The value of the single element of A as a Lisp number (or T/NIL)."
  (unless (= (mlx:size a) 1)
    (error "ITEM needs a single-element array; got shape ~S." (mlx:shape a)))
  (row-major-aref (copy-to-lisp a) 0))

;;; ------------------------------------------------------------------
;;; Printing

(defmethod print-object ((a mlx:mlx-array) stream)
  (cond ((mlx:freed-p a)
         (print-unreadable-object (a stream :type t) (write-string "freed" stream)))
        (*print-readably* (error 'print-not-readable :object a))
        (t
         (let ((text (handler-case (tostring ffi:mlx-array-tostring (ptr a))
                       (error () nil))))
           (print-unreadable-object (a stream :type t)
             (format stream "~(~A~) (~{~D~^ ~})" (mlx:dtype a) (mlx:shape a))
             (when text
               (format stream "~%~A" text)))))))

