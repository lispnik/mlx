;;;; dtype.lisp -- element types

(in-package :mlx.impl)

(defparameter +dtypes+
  ;; keyword     cffi type    lisp element type          kind
  '((:bool       :bool        t                          :bool)
    (:uint8      :uint8       (unsigned-byte 8)          :unsigned)
    (:uint16     :uint16      (unsigned-byte 16)         :unsigned)
    (:uint32     :uint32      (unsigned-byte 32)         :unsigned)
    (:uint64     :uint64      (unsigned-byte 64)         :unsigned)
    (:int8       :int8        (signed-byte 8)            :signed)
    (:int16      :int16       (signed-byte 16)           :signed)
    (:int32      :int32       (signed-byte 32)           :signed)
    (:int64      :int64       (signed-byte 64)           :signed)
    (:float16    nil          single-float               :float)
    (:float32    :float       single-float               :float)
    (:float64    :double      double-float               :float)
    (:bfloat16   nil          single-float               :float)
    (:complex64  nil          (complex single-float)     :complex))
  "Every MLX dtype.  float16/bfloat16 have no CFFI type: they are converted
through float32 when moving data between Lisp and MLX.")

(defun mlx:dtypes ()
  "List of all dtype keywords."
  (mapcar #'first +dtypes+))

(defun dtype-entry (dtype)
  (or (assoc dtype +dtypes+)
      (error "Unknown MLX dtype ~S; expected one of ~S." dtype (mlx:dtypes))))

(defun check-dtype (dtype) (first (dtype-entry dtype)))
(defun dtype-cffi-type (dtype) (second (dtype-entry dtype)))
(defun dtype-lisp-type (dtype) (third (dtype-entry dtype)))
(defun dtype-kind (dtype) (fourth (dtype-entry dtype)))

(defun integer-dtype-p (dtype) (member (dtype-kind dtype) '(:signed :unsigned)))
(defun float-dtype-p (dtype) (eq (dtype-kind dtype) :float))
(defun inexact-dtype-p (dtype) (member (dtype-kind dtype) '(:float :complex)))

(defun mlx:dtype-size (dtype)
  "Size in bytes of one element of DTYPE."
  (ffi:mlx-dtype-size (check-dtype dtype)))

(defun lisp-type-dtype (element-type)
  "The dtype matching a specialized Lisp array ELEMENT-TYPE, or NIL."
  (cond ((subtypep element-type 'nil) nil)
        ((subtypep element-type 'single-float) :float32)
        ((subtypep element-type 'double-float) :float64)
        ((subtypep element-type '(complex single-float)) :complex64)
        ((subtypep element-type 'bit) :bool)
        ((subtypep element-type '(unsigned-byte 8)) :uint8)
        ((subtypep element-type '(unsigned-byte 16)) :uint16)
        ((subtypep element-type '(unsigned-byte 32)) :uint32)
        ((subtypep element-type '(unsigned-byte 64)) :uint64)
        ((subtypep element-type '(signed-byte 8)) :int8)
        ((subtypep element-type '(signed-byte 16)) :int16)
        ((subtypep element-type '(signed-byte 32)) :int32)
        ((subtypep element-type '(signed-byte 64)) :int64)
        (t nil)))

(defun infer-dtype (elements)
  "Dtype for the simple-vector of Lisp scalars ELEMENTS, following MLX/NumPy
defaults: booleans -> bool, integers -> int32 (int64 if needed), reals ->
float32, complexes -> complex64."
  (declare (type simple-vector elements) (optimize speed))
  ;; kind: 0 bool, 1 integer, 2 real, 3 complex
  (let ((kind 0) (big nil))
    (declare (type (integer 0 3) kind))
    (dotimes (i (length elements))
      (let ((x (svref elements i)))
        (typecase x
          (fixnum (when (< kind 1) (setf kind 1))
                  (unless (typep x '(signed-byte 32)) (setf big t)))
          (float (when (< kind 2) (setf kind 2)))
          (integer (when (< kind 1) (setf kind 1)) (setf big t))
          (rational (when (< kind 2) (setf kind 2)))
          (complex (setf kind 3))
          (t (unless (or (eq x t) (null x))
               (error "Cannot put ~S in an MLX array." x))))))
    (case kind
      (0 (if (zerop (length elements)) :float32 :bool))
      (1 (if big :int64 :int32))
      (2 :float32)
      (t :complex64))))

(defun scalar-dtype (x)
  (etypecase x
    ((member t nil) :bool)
    (integer (if (typep x '(signed-byte 32)) :int32 :int64))
    (real :float32)
    (complex :complex64)))

(defun weak-scalar-dtype (x like)
  "MLX's weak typing for Lisp scalars mixed with an array of dtype LIKE:
an integer takes LIKE's dtype unless LIKE is bool; a float takes LIKE's
dtype when LIKE is inexact."
  (let ((kind (dtype-kind like)))
    (etypecase x
      (integer (if (eq kind :bool) (scalar-dtype x) like))
      (real (if (member kind '(:float :complex)) like :float32))
      (complex :complex64)
      ((member t nil) :bool))))

(defun pack-optional-int (x)
  (if x (logior (ldb (byte 32 0) x) (ash 1 32)) 0))

(defun pack-optional-float (x)
  (if x
      (logior (ldb (byte 32 0) (sb-kernel:single-float-bits (float x 1f0))) (ash 1 32))
      0))

(defun pack-optional-dtype (x)
  (if x
      (logior (cffi:foreign-enum-value 'ffi:mlx-dtype (check-dtype x)) (ash 1 32))
      0))
