;;;; device.lisp -- strings, devices and streams

(in-package :mlx.impl)

;;; mlx_string

(defmacro with-mlx-string ((var) &body body)
  "Bind VAR to a pointer to a fresh mlx_string slot; the string is freed after BODY."
  `(cffi:with-foreign-object (,var :pointer)
     (setf (cffi:mem-ref ,var :pointer) (cffi:null-pointer))
     (unwind-protect (progn ,@body)
       (let ((p (cffi:mem-ref ,var :pointer)))
         (unless (cffi:null-pointer-p p) (ffi:mlx-string-free p))))))

(defun mlx-string-value (slot)
  (ffi:mlx-string-data (cffi:mem-ref slot :pointer)))

(defmacro tostring (c-function object)
  "Call an mlx_*_tostring function and return the Lisp string."
  (let ((s (gensym)))
    `(with-mlx-string (,s)
       (check (,c-function ,s ,object))
       (mlx-string-value ,s))))

(defun mlx:version ()
  "The MLX version string."
  (with-mlx-string (s)
    (check (ffi:mlx-version s))
    (mlx-string-value s)))

;;; Devices

(define-handle-type mlx:mlx-device ffi:mlx-device-free "An MLX device (CPU or GPU).")

(defmethod print-object ((d mlx:mlx-device) stream)
  (print-unreadable-object (d stream :type t)
    (if (mlx:freed-p d)
        (write-string "freed" stream)
        (format stream "~(~A~) ~D" (mlx:device-type d) (mlx:device-index d)))))

(defun mlx:make-device (type &optional (index 0))
  "A device handle.  TYPE is :CPU or :GPU."
  (%wrap-mlx-device (ffi:mlx-device-new-type type index)))

(defun mlx:device-type (device)
  (cffi:with-foreign-object (ty 'ffi:mlx-device-type)
    (check (ffi:mlx-device-get-type ty (ptr device)))
    (cffi:mem-ref ty 'ffi:mlx-device-type)))

(defun mlx:device-index (device)
  (cffi:with-foreign-object (i :int)
    (check (ffi:mlx-device-get-index i (ptr device)))
    (cffi:mem-ref i :int)))

(defun coerce-device (x)
  (etypecase x
    (mlx:mlx-device x)
    ((member :cpu :gpu) (mlx:make-device x))))

(defun mlx:default-device ()
  (cffi:with-foreign-object (d :pointer)
    (setf (cffi:mem-ref d :pointer) (cffi:null-pointer))
    (check (ffi:mlx-get-default-device d))
    (%wrap-mlx-device (cffi:mem-ref d :pointer))))

(defun mlx:set-default-device (device)
  "Make DEVICE (a device, :CPU or :GPU) the global default."
  (check (ffi:mlx-set-default-device (ptr (coerce-device device))))
  (reset-stream-cache)
  device)

(defun mlx:device-available-p (device)
  (cffi:with-foreign-object (b :bool)
    (check (ffi:mlx-device-is-available b (ptr (coerce-device device))))
    (cffi:mem-ref b :bool)))

(defun mlx:device-count (type)
  "Number of available devices of TYPE (:CPU or :GPU)."
  (cffi:with-foreign-object (n :int)
    (check (ffi:mlx-device-count n type))
    (cffi:mem-ref n :int)))

(defun mlx:device-info (&optional (device (mlx:default-device)))
  "Properties of DEVICE as a plist of (:key value ...), e.g. :DEVICE-NAME."
  (let ((info (ffi:mlx-device-info-new)))
    (cffi:with-foreign-object (slot :pointer)
      (setf (cffi:mem-ref slot :pointer) info)
      (unwind-protect
           (progn
             (check (ffi:mlx-device-info-get slot (ptr (coerce-device device))))
             (setf info (cffi:mem-ref slot :pointer))
             (let ((keys (ffi:mlx-vector-string-new)))
               (cffi:with-foreign-object (kslot :pointer)
                 (setf (cffi:mem-ref kslot :pointer) keys)
                 (unwind-protect
                      (progn
                        (check (ffi:mlx-device-info-get-keys kslot info))
                        (setf keys (cffi:mem-ref kslot :pointer))
                        (loop for key in (vector-string->list keys)
                              nconc (list (intern (string-upcase (substitute #\- #\_ key)) :keyword)
                                          (device-info-value info key))))
                   (ffi:mlx-vector-string-free keys)))))
        (ffi:mlx-device-info-free info)))))

(defun device-info-value (info key)
  (cffi:with-foreign-objects ((is-string :bool) (s :pointer) (n :size))
    (check (ffi:mlx-device-info-is-string is-string info key))
    (if (cffi:mem-ref is-string :bool)
        (progn (check (ffi:mlx-device-info-get-string s info key))
               (cffi:foreign-string-to-lisp (cffi:mem-ref s :pointer)))
        (progn (check (ffi:mlx-device-info-get-size n info key))
               (cffi:mem-ref n :size)))))

;;; Streams

(define-handle-type mlx:mlx-stream ffi:mlx-stream-free "An MLX stream (an ordered queue of work on a device).")

(defmethod print-object ((s mlx:mlx-stream) stream)
  (print-unreadable-object (s stream :type t)
    (if (mlx:freed-p s)
        (write-string "freed" stream)
        (write-string (tostring ffi:mlx-stream-tostring (ptr s)) stream))))

(defvar mlx:*stream* nil
  "The stream operations run on when no :STREAM argument is given.  NIL
means the default stream of the default device.  May also be a device, or
:CPU / :GPU.  Bind it with WITH-STREAM or WITH-DEVICE.")

(defvar *stream-cache* (make-hash-table :test 'equal)
  "Default streams by device (type . index).  Reset by REINITIALIZE and by
changes made through SET-DEFAULT-DEVICE / SET-DEFAULT-STREAM.")

(defvar *default-stream-cache* nil "Cached stream of the default device.")

(defun reset-stream-cache ()
  (setf *stream-cache* (make-hash-table :test 'equal)
        *default-stream-cache* nil))

(defun mlx:default-stream (&optional device)
  "The default stream of DEVICE (default: the default device)."
  (let ((device (if device (coerce-device device) (mlx:default-device))))
    (cffi:with-foreign-object (s :pointer)
      (setf (cffi:mem-ref s :pointer) (cffi:null-pointer))
      (check (ffi:mlx-get-default-stream s (ptr device)))
      (%wrap-mlx-stream (cffi:mem-ref s :pointer)))))

(defun mlx:set-default-stream (stream)
  (check (ffi:mlx-set-default-stream (ptr stream)))
  (reset-stream-cache)
  stream)

(defun mlx:make-stream (&optional (device (mlx:default-device)))
  "A new stream on DEVICE (a device, :CPU or :GPU)."
  (%wrap-mlx-stream (ffi:mlx-stream-new-device (ptr (coerce-device device)))))

(defun mlx:stream-device (stream)
  (cffi:with-foreign-object (d :pointer)
    (setf (cffi:mem-ref d :pointer) (cffi:null-pointer))
    (check (ffi:mlx-stream-get-device d (ptr stream)))
    (%wrap-mlx-device (cffi:mem-ref d :pointer))))

(defun mlx:stream-index (stream)
  (cffi:with-foreign-object (i :int)
    (check (ffi:mlx-stream-get-index i (ptr stream)))
    (cffi:mem-ref i :int)))

(defun mlx:synchronize (&optional stream)
  "Wait for all work queued on STREAM (default: the current stream) to finish."
  (check (ffi:mlx-synchronize (resolve-stream stream)))
  nil)

(defun cached-device-stream (device)
  (let ((key (cons (mlx:device-type device) (mlx:device-index device))))
    (or (gethash key *stream-cache*)
        (setf (gethash key *stream-cache*) (mlx:default-stream device)))))

(defun resolve-stream (s)
  "Foreign stream pointer for an operation's :STREAM argument S."
  (let ((s (or s mlx:*stream*)))
    (etypecase s
      (null (ptr (or *default-stream-cache*
                     (setf *default-stream-cache* (mlx:default-stream)))))
      (mlx:mlx-stream (ptr s))
      (mlx:mlx-device (ptr (cached-device-stream s)))
      ((member :cpu :gpu)
       (ptr (or (gethash (cons s 0) *stream-cache*)
                (setf (gethash (cons s 0) *stream-cache*)
                      (mlx:default-stream (mlx:make-device s)))))))))

(defmacro mlx:with-stream ((stream) &body body)
  "Run BODY with operations defaulting to STREAM (a stream, device, :CPU or :GPU)."
  `(let ((mlx:*stream* ,stream)) ,@body))

(defmacro mlx:with-device ((device) &body body)
  "Run BODY with operations defaulting to DEVICE's default stream."
  `(let ((mlx:*stream* (coerce-device ,device))) ,@body))
