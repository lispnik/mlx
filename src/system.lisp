;;;; system.lisp -- memory, backends, graph export, distributed groups

(in-package :mlx.impl)

;;; Memory

(defmacro size-query (c-function &rest args)
  `(cffi:with-foreign-object (n :size)
     (check (,c-function n ,@args))
     (cffi:mem-ref n :size)))

(defun mlx:active-memory () "Bytes currently held by live arrays." (size-query ffi:mlx-get-active-memory))
(defun mlx:cache-memory () "Bytes held in the allocator cache." (size-query ffi:mlx-get-cache-memory))
(defun mlx:peak-memory () "Peak bytes allocated since start or reset." (size-query ffi:mlx-get-peak-memory))
(defun mlx:memory-limit () "The memory limit in bytes." (size-query ffi:mlx-get-memory-limit))
(defun mlx:reset-peak-memory () (check (ffi:mlx-reset-peak-memory)) nil)
(defun mlx:clear-cache () "Release cached, unused memory." (check (ffi:mlx-clear-cache)) nil)

(defun mlx:set-memory-limit (bytes)
  "Set the memory limit; returns the previous limit."
  (size-query ffi:mlx-set-memory-limit bytes))

(defun mlx:set-cache-limit (bytes)
  "Set the allocator cache limit; returns the previous limit."
  (size-query ffi:mlx-set-cache-limit bytes))

(defun mlx:set-wired-limit (bytes)
  "Set the wired memory limit (macOS 15+); returns the previous limit."
  (size-query ffi:mlx-set-wired-limit bytes))

;;; Backends

(defun mlx:metal-available-p ()
  (cffi:with-foreign-object (b :bool)
    (check (ffi:mlx-metal-is-available b))
    (cffi:mem-ref b :bool)))

(defun mlx:cuda-available-p ()
  (cffi:with-foreign-object (b :bool)
    (check (ffi:mlx-cuda-is-available b))
    (cffi:mem-ref b :bool)))

(defun mlx:start-metal-capture (path)
  "Start a Metal GPU trace written to PATH (a .gputrace); needs
MTL_CAPTURE_ENABLED=1 in the environment."
  (check (ffi:mlx-metal-start-capture (native-file path)) "start-metal-capture")
  path)

(defun mlx:stop-metal-capture ()
  (check (ffi:mlx-metal-stop-capture))
  nil)

;;; Graph export

(defun call-with-memstream (function)
  "Call FUNCTION with a C FILE* backed by memory; return what it wrote."
  (cffi:with-foreign-objects ((buf :pointer) (len :size))
    (let ((file (cffi:foreign-funcall "open_memstream" :pointer buf :pointer len :pointer)))
      (when (cffi:null-pointer-p file) (error "open_memstream failed"))
      (unwind-protect (funcall function file)
        (cffi:foreign-funcall "fclose" :pointer file :int))
      (let ((p (cffi:mem-ref buf :pointer)))
        (prog1 (cffi:foreign-string-to-lisp p :count (cffi:mem-ref len :size))
          (cffi:foreign-free p))))))

(defun graph-text (c-function outputs names)
  (let ((namer (ffi:mlx-node-namer-new)))
    (unwind-protect
         (progn
           (loop for (array . name) in names
                 do (check (ffi:mlx-node-namer-set-name namer (ptr array) name)))
           (with-vector-array (outs (if (listp outputs) outputs (list outputs)))
             (call-with-memstream
              (lambda (file) (check (funcall c-function file namer outs) "graph export")))))
      (ffi:mlx-node-namer-free namer))))

(defun mlx:export-to-dot (outputs &key names)
  "The computation graph of OUTPUTS (an array or list) in Graphviz DOT
format.  NAMES is an alist of (array . \"label\")."
  (graph-text #'ffi:mlx-export-to-dot outputs names))

(defun mlx:print-graph (outputs &key names)
  "A textual listing of the computation graph of OUTPUTS."
  (graph-text #'ffi:mlx-print-graph outputs names))

;;; Distributed

(define-handle-type mlx-distributed-group ffi:mlx-distributed-group-free
  "A group of processes for distributed communication.")

(defun mlx.distributed:available-p (&optional backend)
  "True if a distributed backend (BACKEND, a string, or any) is available."
  (ffi:mlx-distributed-is-available (or backend (cffi:null-pointer))))

(defun mlx.distributed:init (&key strict backend)
  "Initialize distributed communication and return the global group.  With
STRICT, signal an error if no backend can be initialized (otherwise a
singleton group is returned)."
  (with-out-slots (res)
    (check (ffi:mlx-distributed-init res strict (or backend (cffi:null-pointer)))
           "distributed init")
    (%wrap-mlx-distributed-group (cffi:mem-ref res :pointer))))

(defun mlx.distributed:group-rank (group) (ffi:mlx-distributed-group-rank (ptr group)))
(defun mlx.distributed:group-size (group) (ffi:mlx-distributed-group-size (ptr group)))

(defun mlx.distributed:group-split (group color &key (key -1))
  "Split GROUP into subgroups of processes sharing COLOR, ordered by KEY."
  (with-out-slots (res)
    (check (ffi:mlx-distributed-group-split res (ptr group) color key) "group split")
    (%wrap-mlx-distributed-group (cffi:mem-ref res :pointer))))
