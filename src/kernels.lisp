;;;; kernels.lisp -- custom Metal kernels (mlx.core.fast.metal_kernel)

(in-package :mlx.impl)

(define-handle-type mlx-metal-kernel ffi:mlx-fast-metal-kernel-free)

(defun add-template-arg (config name value)
  (let ((name (if (stringp name) name (symbol-name name))))
    (cond ((and (keywordp value) (member value (mlx:dtypes)))
           (check (ffi:mlx-fast-metal-kernel-config-add-template-arg-dtype config name value)))
          ((integerp value)
           (check (ffi:mlx-fast-metal-kernel-config-add-template-arg-int config name value)))
          ((member value '(t nil))
           (check (ffi:mlx-fast-metal-kernel-config-add-template-arg-bool config name value)))
          (t (error "Template argument ~A must be a dtype, integer or boolean: ~S" name value)))))

(defun triple (x)
  (let ((l (if (listp x) x (list x))))
    (append l (make-list (- 3 (length l)) :initial-element 1))))

(defun mlx.fast:metal-kernel (&key name input-names output-names source (header "")
                                   (ensure-row-contiguous t) atomic-outputs)
  "Build a custom Metal kernel from the body SOURCE (Metal Shading Language).
Inputs and outputs are named by the string lists INPUT-NAMES and
OUTPUT-NAMES.  Returns a function taking keyword arguments

  :INPUTS         list of arrays (or Lisp data)
  :OUTPUT-SHAPES  list of shapes, one per output
  :OUTPUT-DTYPES  list of dtype keywords, one per output
  :GRID           (x y z) grid size (threads); shorter lists pad with 1
  :THREADGROUP    (x y z) threadgroup size
  :TEMPLATE       list of (name . value) where value is a dtype keyword,
                  an integer or a boolean
  :INIT-VALUE     number to initialize the outputs with
  :VERBOSE        print the generated source
  :STREAM

and returning the list of output arrays."
  (let ((kernel (with-vector-string (ins input-names)
                  (with-vector-string (outs output-names)
                    (%wrap-mlx-metal-kernel
                     (without-float-traps
                       (ffi:mlx-fast-metal-kernel-new name ins outs source header
                                                      ensure-row-contiguous atomic-outputs)))))))
    (lambda (&key inputs output-shapes output-dtypes (grid '(1 1 1)) (threadgroup '(1 1 1))
               template init-value verbose stream)
      (let ((config (ffi:mlx-fast-metal-kernel-config-new)))
        (unwind-protect
             (progn
               (loop for shape in output-shapes
                     for dtype in output-dtypes
                     do (with-int-buffer (p n shape)
                          (check (ffi:mlx-fast-metal-kernel-config-add-output-arg
                                  config p n (check-dtype dtype)))))
               (destructuring-bind (x y z) (triple grid)
                 (check (ffi:mlx-fast-metal-kernel-config-set-grid config x y z)))
               (destructuring-bind (x y z) (triple threadgroup)
                 (check (ffi:mlx-fast-metal-kernel-config-set-thread-group config x y z)))
               (when init-value
                 (check (ffi:mlx-fast-metal-kernel-config-set-init-value config (float init-value 1f0))))
               (check (ffi:mlx-fast-metal-kernel-config-set-verbose config verbose))
               (loop for (tname . tvalue) in template do (add-template-arg config tname tvalue))
               (with-vector-array (in inputs)
                 (with-out-slots (res)
                   (check (ffi:mlx-fast-metal-kernel-apply res (ptr kernel) in config
                                                           (resolve-stream stream))
                          "metal kernel")
                   (vector-array->list (cffi:mem-ref res :pointer) :free t))))
          (ffi:mlx-fast-metal-kernel-config-free config))))))
