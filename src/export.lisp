;;;; export.lisp -- exporting traced functions to files and importing them
;;;;
;;;; Keyword arguments cross the boundary as name -> array maps: a Lisp
;;;; call (f x y :bias b) passes {"bias": b}; an exported Lisp function
;;;; receives them back as keyword arguments (:BIAS b).

(in-package :mlx.impl)

(defun split-kwargs (args)
  "Split ARGS at the first keyword into (values positional kwargs-alist)."
  (let ((k (position-if #'keywordp args)))
    (if k
        (values (subseq args 0 k)
                (loop for (key value) on (subseq args k) by #'cddr
                      collect (cons (string-downcase (symbol-name key)) value)))
        (values args nil))))

(defun kwargs-as-keys (alist)
  (loop for (k . v) in alist
        nconc (list (intern (string-upcase k) :keyword) v)))

(defun kwargs-closure (function)
  (make-closure-kwargs
   (lambda (inputs kwargs)
     (apply function (append inputs (kwargs-as-keys kwargs))))))

(defun mlx:export-function (file function &rest example-args)
  "Trace FUNCTION on EXAMPLE-ARGS (arrays, optionally followed by keyword
arguments) and write the graph to FILE, loadable with IMPORT-FUNCTION from
Lisp, Python, C++ or Swift.  A keyword argument :SHAPELESS T among the
example args allows later calls with other shapes."
  (multiple-value-bind (args kwargs) (split-kwargs example-args)
    (let* ((shapeless (cdr (assoc "shapeless" kwargs :test #'string=)))
           (kwargs (remove "shapeless" kwargs :key #'car :test #'string=)))
      (with-vector-array (in args)
        (if kwargs
            (let ((closure (kwargs-closure function))
                  (map (make-map-string-to-array kwargs)))
              (unwind-protect
                   (check (ffi:mlx-export-function-kwargs (native-file file) (ptr closure) in map
                                                          shapeless)
                          "export-function")
                (ffi:mlx-map-string-to-array-free map)
                (mlx:free closure)))
            (let ((closure (flat-closure function)))
              (unwind-protect
                   (check (ffi:mlx-export-function (native-file file) (ptr closure) in shapeless)
                          "export-function")
                (mlx:free closure)))))))
  file)

(define-handle-type mlx-imported-function ffi:mlx-imported-function-free)

(defun mlx:import-function (file)
  "Load a function exported by EXPORT-FUNCTION (or by MLX in any language).
Returns a Lisp function of arrays (optionally followed by keyword
arguments) that returns the list of output arrays."
  (let ((p (without-float-traps (ffi:mlx-imported-function-new (native-file file)))))
    (when (cffi:null-pointer-p p) (signal-mlx-error "import-function"))
    (let ((handle (mlx:persist (%wrap-mlx-imported-function p))))
      (lambda (&rest args)
        (multiple-value-bind (positional kwargs) (split-kwargs args)
          (with-vector-array (in positional)
            (with-out-slots (res)
              (if kwargs
                  (let ((map (make-map-string-to-array kwargs)))
                    (unwind-protect
                         (check (ffi:mlx-imported-function-apply-kwargs res (ptr handle) in map)
                                "imported function")
                      (ffi:mlx-map-string-to-array-free map)))
                  (check (ffi:mlx-imported-function-apply res (ptr handle) in)
                         "imported function"))
              (vector-array->list (cffi:mem-ref res :pointer) :free t))))))))

(define-handle-type mlx-function-exporter ffi:mlx-function-exporter-free)

(defmacro mlx:with-function-exporter ((trace file function &key shapeless) &body body)
  "Export several traces of FUNCTION (of positional arrays) to one FILE.
Within BODY, TRACE names a local function: (TRACE args...) records a trace
for those example arguments."
  (let ((exporter (gensym "EXPORTER")) (closure (gensym "CLOSURE")))
    `(let* ((,closure (flat-closure ,function))
            (,exporter (%wrap-mlx-function-exporter
                        (without-float-traps
                          (ffi:mlx-function-exporter-new (native-file ,file) (ptr ,closure)
                                                         ,shapeless)))))
       (unwind-protect
            (flet ((,trace (&rest args) (exporter-trace ,exporter args)))
              ,@body)
         (mlx:free ,exporter ,closure)))))

(defun exporter-trace (exporter args)
  (with-vector-array (in args)
    (check (ffi:mlx-function-exporter-apply (ptr exporter) in) "function exporter")))
