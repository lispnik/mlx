;;;; vector.lisp -- mlx_vector_* and mlx_map_* <-> Lisp sequences and tables

(in-package :mlx.impl)

;;; vector_string

(defun vector-string->list (vec)
  (loop for i below (ffi:mlx-vector-string-size vec)
        collect (cffi:with-foreign-object (s :pointer)
                  (check (ffi:mlx-vector-string-get s vec i))
                  (cffi:foreign-string-to-lisp (cffi:mem-ref s :pointer)))))

(defun list->vector-string (strings)
  "A new mlx_vector_string holding STRINGS (caller frees)."
  (let ((vec (ffi:mlx-vector-string-new)))
    (dolist (s strings vec)
      (check (ffi:mlx-vector-string-append-value vec (string s))))))

(defmacro with-vector-string ((var strings) &body body)
  `(let ((,var (list->vector-string ,strings)))
     (unwind-protect (progn ,@body) (ffi:mlx-vector-string-free ,var))))

;;; vector_int

(defun vector-int->list (vec)
  (cffi:with-foreign-object (x :int)
    (loop for i below (ffi:mlx-vector-int-size vec)
          collect (progn (check (ffi:mlx-vector-int-get x vec i))
                         (cffi:mem-ref x :int)))))

(defun list->vector-int (ints)
  (let ((vec (ffi:mlx-vector-int-new)))
    (dolist (i ints vec)
      (check (ffi:mlx-vector-int-append-value vec i)))))

;;; vector_array

(defun vector-array->list (vec &key free)
  "Wrap the elements of the mlx_vector_array VEC as fresh MLX-ARRAYs.
With FREE, also release VEC."
  (unwind-protect
       (loop for i below (ffi:mlx-vector-array-size vec)
             collect (cffi:with-foreign-object (slot :pointer)
                       (setf (cffi:mem-ref slot :pointer) (cffi:null-pointer))
                       (check (ffi:mlx-vector-array-get slot vec i))
                       (%wrap-mlx-array (cffi:mem-ref slot :pointer))))
    (when free (ffi:mlx-vector-array-free vec))))

(defun list->vector-array (arrays)
  "A new mlx_vector_array holding ARRAYS (arrays or Lisp data; caller frees)."
  (let ((vec (ffi:mlx-vector-array-new)))
    (dolist (a arrays vec)
      (with-array-arg (p a)
        (check (ffi:mlx-vector-array-append-value vec p))))))

(defmacro with-vector-array ((var arrays) &body body)
  "Bind VAR to a temporary mlx_vector_array of ARRAYS (a list, or one array)."
  (let ((a (gensym)))
    `(let* ((,a ,arrays)
            (,var (list->vector-array (if (listp ,a) ,a (list ,a)))))
       (unwind-protect (progn ,@body) (ffi:mlx-vector-array-free ,var)))))

(defun set-vector-array-out (slot arrays)
  "Store ARRAYS into the mlx_vector_array pointed to by SLOT (a callback's
result argument)."
  (with-vector-array (v arrays)
    (check (ffi:mlx-vector-array-set slot v))))

;;; maps
;;;
;;; Iterators are two-word structs {ctx, map_ctx}.  The constructor returns
;;; ctx in x0 (what we receive) and map_ctx -- always the map's own ctx --
;;; in x1, so we pass the map pointer back as the second word.

(defun map-string-to-array->alist (map)
  (let ((it (ffi:mlx-map-string-to-array-iterator-new map)))
    (unwind-protect
         (cffi:with-foreign-objects ((key :pointer) (value :pointer))
           (loop do (setf (cffi:mem-ref value :pointer) (cffi:null-pointer))
                 while (zerop (without-float-traps
                                (ffi:mlx-map-string-to-array-iterator-next key value it map)))
                 collect (cons (cffi:foreign-string-to-lisp (cffi:mem-ref key :pointer))
                               (%wrap-mlx-array (cffi:mem-ref value :pointer)))))
      (ffi:mlx-map-string-to-array-iterator-free it map))))

(defun map-string-to-string->alist (map)
  (let ((it (ffi:mlx-map-string-to-string-iterator-new map)))
    (unwind-protect
         (cffi:with-foreign-objects ((key :pointer) (value :pointer))
           (loop while (zerop (ffi:mlx-map-string-to-string-iterator-next key value it map))
                 collect (cons (cffi:foreign-string-to-lisp (cffi:mem-ref key :pointer))
                               (cffi:foreign-string-to-lisp (cffi:mem-ref value :pointer)))))
      (ffi:mlx-map-string-to-string-iterator-free it map))))

(defun alist->hash-table (alist)
  (let ((h (make-hash-table :test 'equal)))
    (loop for (k . v) in alist do (setf (gethash k h) v))
    h))

(defun mapping->alist (mapping)
  "Normalize a hash table, alist or plist with string/symbol keys to an
alist with string keys.  Symbol keys are downcased."
  (flet ((key (k) (if (stringp k) k (string-downcase (string k)))))
    (etypecase mapping
      (null nil)
      (hash-table (let ((out '()))
                    (maphash (lambda (k v) (push (cons (key k) v) out)) mapping)
                    (sort out #'string< :key #'car)))
      (cons (if (consp (first mapping))
                (mapcar (lambda (e) (cons (key (car e)) (cdr e))) mapping)
                (loop for (k v) on mapping by #'cddr collect (cons (key k) v)))))))

(defun make-map-string-to-array (mapping)
  (let ((map (ffi:mlx-map-string-to-array-new)))
    (loop for (k . v) in (mapping->alist mapping)
          do (with-array-arg (p v)
               (check (ffi:mlx-map-string-to-array-insert map k p))))
    map))

(defun make-map-string-to-string (mapping)
  (let ((map (ffi:mlx-map-string-to-string-new)))
    (loop for (k . v) in (mapping->alist mapping)
          do (check (ffi:mlx-map-string-to-string-insert map k (princ-to-string v))))
    map))

;;; ------------------------------------------------------------------
;;; Evaluation

(defun collect-arrays (tree)
  (remove-if-not (lambda (h) (typep h 'mlx:mlx-array)) (reverse (collect-handles tree))))

(defun mlx:eval (&rest arrays)
  "Evaluate ARRAYS (arrays or trees of arrays) and return the first argument."
  (with-vector-array (v (collect-arrays arrays))
    (check (ffi:mlx-eval v) "eval"))
  (first arrays))

(defun mlx:async-eval (&rest arrays)
  "Start evaluating ARRAYS in the background and return the first argument."
  (with-vector-array (v (collect-arrays arrays))
    (check (ffi:mlx-async-eval v) "async-eval"))
  (first arrays))
