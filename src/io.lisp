;;;; io.lisp -- loading and saving arrays (.npy, .safetensors, .gguf),
;;;; to files or to in-memory octet vectors

(in-package :mlx.impl)

(defun native-file (file) (uiop:native-namestring (merge-pathnames file)))

(defun load-stream (stream)
  "Loading is a CPU operation in MLX; default to the CPU stream."
  (resolve-stream (or stream :cpu)))

(defun mlx:save (file array)
  "Save ARRAY to FILE in NumPy .npy format."
  (with-array-arg (p array)
    (check (ffi:mlx-save (native-file file) p) "save"))
  file)

(defun mlx:load (file &key stream)
  "Load FILE, choosing the format by extension:
  .npy          -> an array
  .safetensors  -> (values hash-table-of-arrays metadata-hash-table)
  .gguf         -> (values hash-table-of-arrays gguf-handle)  see GGUF-METADATA"
  (let ((type (string-downcase (or (pathname-type (pathname file)) ""))))
    (cond ((string= type "safetensors") (mlx:load-safetensors file :stream stream))
          ((string= type "gguf") (mlx:load-gguf file :stream stream))
          (t (with-out-slots (res)
               (check (ffi:mlx-load res (native-file file) (load-stream stream)) "load")
               (%wrap-mlx-array (cffi:mem-ref res :pointer)))))))

(defun read-safetensors-result (arrays-slot metadata-slot)
  (let ((arrays (cffi:mem-ref arrays-slot :pointer))
        (metadata (cffi:mem-ref metadata-slot :pointer)))
    (unwind-protect
         (values (alist->hash-table (map-string-to-array->alist arrays))
                 (alist->hash-table (map-string-to-string->alist metadata)))
      (ffi:mlx-map-string-to-array-free arrays)
      (ffi:mlx-map-string-to-string-free metadata))))

(defun mlx:load-safetensors (file &key stream)
  "Returns (values arrays metadata): hash tables keyed by name."
  (with-out-slots (arrays metadata)
    (setf (cffi:mem-ref arrays :pointer) (ffi:mlx-map-string-to-array-new)
          (cffi:mem-ref metadata :pointer) (ffi:mlx-map-string-to-string-new))
    (check (ffi:mlx-load-safetensors arrays metadata (native-file file) (load-stream stream))
           "load-safetensors")
    (read-safetensors-result arrays metadata)))

(defmacro with-safetensors-maps ((amap mmap arrays metadata) &body body)
  `(let ((,amap (make-map-string-to-array ,arrays))
         (,mmap (make-map-string-to-string ,metadata)))
     (unwind-protect (progn ,@body)
       (ffi:mlx-map-string-to-array-free ,amap)
       (ffi:mlx-map-string-to-string-free ,mmap))))

(defun mlx:save-safetensors (file arrays &key metadata)
  "Save ARRAYS -- a hash table, alist or plist of name -> array -- to FILE.
METADATA is a mapping of string keys to (printed) string values."
  (with-safetensors-maps (amap mmap arrays metadata)
    (check (ffi:mlx-save-safetensors (native-file file) amap mmap) "save-safetensors"))
  file)

;;; GGUF

(define-handle-type mlx-gguf ffi:mlx-io-gguf-free "A loaded GGUF file.")

(defun mlx:load-gguf (file &key stream)
  "Returns (values arrays gguf): a hash table of the tensors, and a handle
for reading metadata with GGUF-METADATA."
  (with-out-slots (res)
    (check (ffi:mlx-load-gguf res (native-file file) (load-stream stream)) "load-gguf")
    (let* ((gguf (%wrap-mlx-gguf (cffi:mem-ref res :pointer)))
           (table (make-hash-table :test 'equal)))
      (with-out-slots (keys)
        (check (ffi:mlx-io-gguf-get-keys keys (ptr gguf)))
        (let ((vec (cffi:mem-ref keys :pointer)))
          (unwind-protect
               (dolist (k (vector-string->list vec))
                 (with-out-slots (arr)
                   (check (ffi:mlx-io-gguf-get-array arr (ptr gguf) k))
                   (setf (gethash k table) (%wrap-mlx-array (cffi:mem-ref arr :pointer)))))
            (ffi:mlx-vector-string-free vec))))
      (values table gguf))))

(defun mlx:gguf-metadata (gguf key)
  "The metadata value for KEY in GGUF: a string, a list of strings, an
array, or NIL when absent."
  (flet ((has (fn)
           ;; these report failure (not false) for absent keys
           (cffi:with-foreign-object (flag :bool)
             (setf (cffi:mem-ref flag :bool) nil)
             (prog1 (and (zerop (without-float-traps (funcall fn flag (ptr gguf) key)))
                         (cffi:mem-ref flag :bool))
               (setf *last-error-message* nil)))))
    (cond ((has #'ffi:mlx-io-gguf-has-metadata-string)
           (with-mlx-string (s)
             (check (ffi:mlx-io-gguf-get-metadata-string s (ptr gguf) key))
             (mlx-string-value s)))
          ((has #'ffi:mlx-io-gguf-has-metadata-vector-string)
           (with-out-slots (v)
             (check (ffi:mlx-io-gguf-get-metadata-vector-string v (ptr gguf) key))
             (let ((vec (cffi:mem-ref v :pointer)))
               (unwind-protect (vector-string->list vec) (ffi:mlx-vector-string-free vec)))))
          ((has #'ffi:mlx-io-gguf-has-metadata-array)
           (with-out-slots (a)
             (check (ffi:mlx-io-gguf-get-metadata-array a (ptr gguf) key))
             (%wrap-mlx-array (cffi:mem-ref a :pointer))))
          (t nil))))

(defun mlx:save-gguf (file arrays &key metadata)
  "Save ARRAYS (a mapping of name -> array) to FILE in GGUF format.
METADATA maps keys to strings, lists of strings, or arrays/numbers."
  (let ((gguf (ffi:mlx-io-gguf-new)))
    (unwind-protect
         (progn
           (loop for (k . v) in (mapping->alist arrays)
                 do (with-array-arg (p v) (check (ffi:mlx-io-gguf-set-array gguf k p))))
           (loop for (k . v) in (mapping->alist metadata)
                 do (etypecase v
                      (string (check (ffi:mlx-io-gguf-set-metadata-string gguf k v)))
                      ((cons string)
                       (with-vector-string (vs v)
                         (check (ffi:mlx-io-gguf-set-metadata-vector-string gguf k vs))))
                      (t (with-array-arg (p v)
                           (check (ffi:mlx-io-gguf-set-metadata-array gguf k p))))))
           (check (ffi:mlx-save-gguf (native-file file) gguf) "save-gguf"))
      (ffi:mlx-io-gguf-free gguf)))
  file)

;;; ------------------------------------------------------------------
;;; In-memory I/O through the mlx_io_vtable
;;;
;;; The descriptor is a callback-registry id for an OCTET-IO.  MLX may call
;;; these from its own threads (loads are lazy), so every callback traps
;;; Lisp errors and marks the stream bad instead of unwinding into C.

(defstruct (octet-io (:constructor %make-octet-io))
  (data (make-array 0 :element-type '(unsigned-byte 8)) :type (simple-array (unsigned-byte 8) (*)))
  (length 0 :type fixnum)
  (pos 0 :type fixnum)
  (good t)
  (label (cffi:null-pointer)))

(defmacro with-io ((var desc) &body body)
  `(let ((,var (callback-object ,desc))) ,@body))

(defun io-ensure-capacity (io needed)
  (let ((data (octet-io-data io)))
    (when (> needed (length data))
      (let ((new (make-array (max needed (* 2 (length data)) 256) :element-type '(unsigned-byte 8))))
        (replace new data :end2 (octet-io-length io))
        (setf (octet-io-data io) new)))))

(defun memcpy (dst src n)
  (cffi:foreign-funcall "memcpy" :pointer dst :pointer src :size n :pointer))

(defun io-read-into (io dest n offset)
  (if (> (+ offset n) (octet-io-length io))
      (setf (octet-io-good io) nil)
      (cffi:with-pointer-to-vector-data (src (octet-io-data io))
        (memcpy dest (cffi:inc-pointer src offset) n))))

(cffi:defcallback io-is-open :bool ((desc :pointer))
  (declare (ignore desc))
  t)

(cffi:defcallback io-good :bool ((desc :pointer))
  (handler-case (with-io (io desc) (octet-io-good io)) (error () nil)))

(cffi:defcallback io-tell :size ((desc :pointer))
  (handler-case (with-io (io desc) (octet-io-pos io)) (error () 0)))

(cffi:defcallback io-seek :void ((desc :pointer) (offset :int64) (whence :int))
  (handler-case
      (with-io (io desc)
        (let ((pos (+ offset (case whence
                               (0 0)
                               (1 (octet-io-pos io))
                               (2 (octet-io-length io))
                               (t 0)))))
          (if (<= 0 pos) (setf (octet-io-pos io) pos) (setf (octet-io-good io) nil))))
    (error () nil)))

(cffi:defcallback io-read :void ((desc :pointer) (data :pointer) (n :size))
  (handler-case
      (with-io (io desc)
        (io-read-into io data n (octet-io-pos io))
        (incf (octet-io-pos io) n))
    (error () nil)))

(cffi:defcallback io-read-at-offset :void ((desc :pointer) (data :pointer) (n :size) (offset :size))
  (handler-case (with-io (io desc) (io-read-into io data n offset))
    (error () nil)))

(cffi:defcallback io-write :void ((desc :pointer) (data :pointer) (n :size))
  (handler-case
      (with-io (io desc)
        (let ((end (+ (octet-io-pos io) n)))
          (io-ensure-capacity io end)
          (cffi:with-pointer-to-vector-data (dst (octet-io-data io))
            (memcpy (cffi:inc-pointer dst (octet-io-pos io)) data n))
          (setf (octet-io-pos io) end
                (octet-io-length io) (max end (octet-io-length io)))))
    (error () nil)))

(cffi:defcallback io-label :pointer ((desc :pointer))
  (handler-case (with-io (io desc) (octet-io-label io)) (error () (cffi:null-pointer))))

(cffi:defcallback io-free :void ((desc :pointer))
  (handler-case
      (with-io (io desc)
        (cffi:foreign-string-free (octet-io-label io))
        (unregister-callback desc))
    (error () nil)))

(defun call-with-vtable (function)
  "Call FUNCTION with a pointer to a filled mlx_io_vtable (valid only
during the call; the C side copies it)."
  (cffi:with-foreign-object (vt '(:struct ffi:mlx-io-vtable))
    (macrolet ((slot (name cb)
                 `(setf (cffi:foreign-slot-value vt '(:struct ffi:mlx-io-vtable) ',name)
                        (cffi:callback ,cb))))
      (slot ffi::is-open io-is-open)
      (slot ffi::good io-good)
      (slot ffi::tell io-tell)
      (slot ffi::seek io-seek)
      (slot ffi::read io-read)
      (slot ffi::read-at-offset io-read-at-offset)
      (slot ffi::write io-write)
      (slot ffi::label io-label)
      (slot ffi::free io-free))
    (funcall function vt)))

(define-handle-type mlx-io-reader ffi:mlx-io-reader-free)
(define-handle-type mlx-io-writer ffi:mlx-io-writer-free)

(defun make-octet-io (&optional octets)
  (let ((data (if octets
                  (coerce octets '(simple-array (unsigned-byte 8) (*)))
                  (make-array 0 :element-type '(unsigned-byte 8)))))
    (%make-octet-io :data data :length (length data)
                   :label (cffi:foreign-string-alloc "<lisp octets>"))))

(defun octet-reader (octets)
  (let ((desc (register-callback (make-octet-io octets))))
    (call-with-vtable (lambda (vt) (%wrap-mlx-io-reader (ffi:mlx-io-reader-new desc vt))))))

(defun octet-writer ()
  "Returns (values writer-handle octet-io)."
  (let* ((io (make-octet-io))
         (desc (register-callback io)))
    (values (call-with-vtable (lambda (vt) (%wrap-mlx-io-writer (ffi:mlx-io-writer-new desc vt))))
            io)))

(defun octet-io-contents (io)
  (subseq (octet-io-data io) 0 (octet-io-length io)))

(defun mlx:save-to-octets (array)
  "The .npy serialization of ARRAY as an (unsigned-byte 8) vector."
  (multiple-value-bind (writer io) (octet-writer)
    (unwind-protect
         (progn
           (with-array-arg (p array)
             (check (ffi:mlx-save-writer (ptr writer) p) "save-to-octets"))
           (octet-io-contents io))
      (mlx:free writer))))

(defun mlx:load-from-octets (octets &key stream)
  "Load an array from .npy OCTETS.  The result is evaluated before return."
  (let ((reader (octet-reader octets)))
    (unwind-protect
         (with-out-slots (res)
           (check (ffi:mlx-load-reader res (ptr reader) (load-stream stream)) "load-from-octets")
           (mlx:eval (%wrap-mlx-array (cffi:mem-ref res :pointer))))
      (mlx:free reader))))

(defun mlx:save-safetensors-to-octets (arrays &key metadata)
  "The .safetensors serialization of ARRAYS (a mapping of name -> array)."
  (multiple-value-bind (writer io) (octet-writer)
    (unwind-protect
         (with-safetensors-maps (amap mmap arrays metadata)
           (check (ffi:mlx-save-safetensors-writer (ptr writer) amap mmap)
                  "save-safetensors-to-octets")
           (octet-io-contents io))
      (mlx:free writer))))

(defun mlx:load-safetensors-from-octets (octets &key stream)
  "Returns (values arrays metadata) from .safetensors OCTETS.  The arrays
are evaluated before return."
  (let ((reader (octet-reader octets)))
    (unwind-protect
         (with-out-slots (arrays metadata)
           (setf (cffi:mem-ref arrays :pointer) (ffi:mlx-map-string-to-array-new)
                 (cffi:mem-ref metadata :pointer) (ffi:mlx-map-string-to-string-new))
           (check (ffi:mlx-load-safetensors-reader arrays metadata (ptr reader)
                                                   (load-stream stream))
                  "load-safetensors-from-octets")
           (multiple-value-bind (table meta) (read-safetensors-result arrays metadata)
             (mlx:eval (loop for v being the hash-values of table collect v))
             (values table meta)))
      (mlx:free reader))))
