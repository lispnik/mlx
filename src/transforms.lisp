;;;; transforms.lisp -- automatic differentiation, vectorization, compilation

(in-package :mlx.impl)

(defun argument-leaf-ranges (args)
  "For each argument tree in ARGS, (start . count) of its leaves in the
flattened argument list."
  (let ((start 0))
    (mapcar (lambda (a)
              (let ((n (leaf-count a)))
                (prog1 (cons start n) (incf start n))))
            args)))

(defun mlx:value-and-grad (function &key (argnums 0))
  "Return a function computing (values (FUNCTION args...) gradients).
FUNCTION's first output leaf must be a scalar (the loss); it may return
any tree, e.g. (list loss aux).  Arguments may be arrays, numbers or trees
of them.  ARGNUMS (an integer or list) selects the arguments to
differentiate; the gradients have the same tree shape as those arguments
-- one tree for an integer ARGNUMS, a list of trees for a list."
  (lambda (&rest args)
    (let* ((argnum-list (if (listp argnums) argnums (list argnums)))
           (ranges (argument-leaf-ranges args))
           (grad-indices
             (loop for n in argnum-list
                   for (start . count) = (or (nth n ranges)
                                             (error "ARGNUMS ~S out of range for ~D arguments."
                                                    argnums (length args)))
                   nconc (loop for i from start below (+ start count) collect i)))
           (tc (make-tree-closure function)))
      (multiple-value-bind (inputs structure) (flatten-args args)
        (setf (tree-closure-in-structure tc) structure)
        (unwind-protect
             (with-out-slots (vag values-slot grads-slot)
               (with-int-buffer (idx n grad-indices)
                 (check (ffi:mlx-value-and-grad vag (ptr (tree-closure-closure tc)) idx n)
                        "value-and-grad"))
               (let ((vag (%wrap-mlx-closure-value-and-grad (cffi:mem-ref vag :pointer))))
                 (unwind-protect
                      (with-vector-array (in inputs)
                        (check (ffi:mlx-closure-value-and-grad-apply values-slot grads-slot
                                                                     (ptr vag) in)
                               "value-and-grad")
                        (let* ((vals (vector-array->list (cffi:mem-ref values-slot :pointer) :free t))
                               (grads (vector-array->list (cffi:mem-ref grads-slot :pointer) :free t))
                               (trees (loop for n in argnum-list
                                            for count = (cdr (nth n ranges))
                                            collect (mlx:tree-unflatten
                                                     (nth-value 1 (mlx:tree-flatten (nth n args)))
                                                     (loop repeat count collect (pop grads))))))
                          (values (first (mlx:tree-unflatten (tree-closure-out-structure tc) vals))
                                  (if (listp argnums) trees (first trees)))))
                   (mlx:free vag))))
          (mlx:free (tree-closure-closure tc)))))))

(defun mlx:grad (function &key (argnums 0))
  "Return a function computing the gradient of FUNCTION (whose first output
must be a scalar) with respect to the arguments selected by ARGNUMS.
See VALUE-AND-GRAD."
  (let ((vg (mlx:value-and-grad function :argnums argnums)))
    (lambda (&rest args)
      (nth-value 1 (apply vg args)))))

(defun flat-closure (function)
  "A closure applying FUNCTION to the input arrays as separate arguments."
  (make-closure (lambda (inputs) (apply function inputs))))

(defun mlx:vjp (function primals cotangents)
  "Vector-Jacobian product.  FUNCTION takes the arrays PRIMALS as arguments
and returns an array or list of arrays; COTANGENTS matches its outputs.
Returns (values outputs vjps), both lists."
  (let ((closure (flat-closure function)))
    (unwind-protect
         (with-vector-array (p primals)
           (with-vector-array (c cotangents)
             (with-out-slots (outs vjps)
               (check (ffi:mlx-vjp outs vjps (ptr closure) p c) "vjp")
               (values (vector-array->list (cffi:mem-ref outs :pointer) :free t)
                       (vector-array->list (cffi:mem-ref vjps :pointer) :free t)))))
      (mlx:free closure))))

(defun mlx:jvp (function primals tangents)
  "Jacobian-vector product.  FUNCTION takes the arrays PRIMALS as arguments;
TANGENTS matches PRIMALS.  Returns (values outputs jvps), both lists."
  (let ((closure (flat-closure function)))
    (unwind-protect
         (with-vector-array (p primals)
           (with-vector-array (tg tangents)
             (with-out-slots (outs jvps)
               (check (ffi:mlx-jvp outs jvps (ptr closure) p tg) "jvp")
               (values (vector-array->list (cffi:mem-ref outs :pointer) :free t)
                       (vector-array->list (cffi:mem-ref jvps :pointer) :free t)))))
      (mlx:free closure))))

(defun normalize-axes (axes count)
  (let ((axes (if (listp axes) axes (make-list count :initial-element axes))))
    (unless (= (length axes) count)
      (error "Expected ~D axes, got ~S." count axes))
    (mapcar (lambda (a) (or a -1)) axes)))

(defun mlx:vmap (function &key (in-axes 0) (out-axes 0))
  "Vectorize FUNCTION (taking and returning arrays) over the given axes.
IN-AXES is an integer, or a list with one entry per argument (NIL: that
argument is not mapped).  OUT-AXES likewise for the outputs.  The result
returns an array when FUNCTION does, else a list."
  (lambda (&rest args)
    (let* ((inputs (mapcar #'mlx:ensure-array args))
           (in-axes (normalize-axes in-axes (length inputs)))
           (single-output nil)
           (closure (make-closure (lambda (xs)
                                    (let ((r (apply function xs)))
                                      (setf single-output (typep r 'mlx:mlx-array))
                                      r)))))
      (unwind-protect
           (with-vector-array (in inputs)
             (with-out-slots (trace-in trace-out res)
               (with-int-buffer (ia nia in-axes)
                 (check (ffi:mlx-detail-vmap-trace trace-in trace-out (ptr closure) in ia nia)
                        "vmap")
                 (let* ((tin (cffi:mem-ref trace-in :pointer))
                        (tout (cffi:mem-ref trace-out :pointer)))
                   (unwind-protect
                        (with-int-buffer (oa noa (normalize-axes out-axes (ffi:mlx-vector-array-size tout)))
                          (check (ffi:mlx-detail-vmap-replace res in tin tout ia nia oa noa) "vmap")
                          (let ((outs (vector-array->list (cffi:mem-ref res :pointer) :free t)))
                            (if single-output (first outs) outs)))
                     (ffi:mlx-vector-array-free tin)
                     (ffi:mlx-vector-array-free tout))))))
        (mlx:free closure)))))

;;; Closure-returning transforms

(defun tree-closure-caller (tc closure-handle)
  "A Lisp function applying CLOSURE-HANDLE (derived from the tree closure TC)
to tree arguments and rebuilding the result tree."
  (lambda (&rest args)
    (multiple-value-bind (inputs structure) (flatten-args args)
      (setf (tree-closure-in-structure tc) structure)
      ;; applying may trace FUNCTION, which records the output skeleton
      (let ((outputs (apply-closure closure-handle inputs)))
        (values-list (mlx:tree-unflatten (tree-closure-out-structure tc) outputs))))))

(defvar *compile-id-cell* (list 0)
  "Counter giving each compiled function a unique cache id.")

(defun mlx:compile (function &key shapeless)
  "Return a compiled version of FUNCTION: the graph traced on the first call
(and whenever input shapes or dtypes change, unless SHAPELESS) is optimized
and fused, and later calls skip Lisp entirely.  FUNCTION should be pure and
take only arrays (or trees of arrays); numbers become array inputs."
  (let* ((tc (make-tree-closure function))
         (fun-id (sb-ext:atomic-incf (car *compile-id-cell*)))
         (compiled (with-out-slots (res)
                     (with-int-buffer (constants n '() :type :uint64)
                       (check (ffi:mlx-detail-compile res (ptr (tree-closure-closure tc))
                                                      fun-id shapeless constants n)
                              "compile"))
                     (cffi:mem-ref res :pointer)))
         (handle (%make-mlx-closure
                  (cons compiled (lambda (p)
                                   (ffi:mlx-closure-free p)
                                   (ffi:mlx-detail-compile-erase fun-id))))))
    (register-handle handle)
    ;; owned by the returned function, whatever scope we are in
    (mlx:persist handle (tree-closure-closure tc))
    (tree-closure-caller tc handle)))

(defun mlx:checkpoint (function)
  "Return FUNCTION with gradient checkpointing: its intermediate values are
recomputed during the backward pass instead of being stored."
  (let ((tc (make-tree-closure function)))
    (with-out-slots (res)
      (check (ffi:mlx-checkpoint res (ptr (tree-closure-closure tc))) "checkpoint")
      (let ((handle (%wrap-mlx-closure (cffi:mem-ref res :pointer))))
        (mlx:persist handle (tree-closure-closure tc))
        (tree-closure-caller tc handle)))))

(defun mlx:custom-function (function &key vjp jvp vmap)
  "Return FUNCTION with custom transformation rules:
  VJP  (lambda (primals cotangents outputs)) -> list of cotangents, one per primal
  JVP  (lambda (primals tangents argnums))  -> list of output tangents
  VMAP (lambda (inputs axes)) -> (values outputs out-axes)
All arguments are lists of arrays."
  (let ((tc (make-tree-closure function))
        (vjp-c (and vjp (make-closure-custom vjp)))
        (jvp-c (and jvp (make-closure-custom-jvp jvp)))
        (vmap-c (and vmap (make-closure-custom-vmap vmap))))
    (flet ((p (h) (if h (ptr h) (cffi:null-pointer))))
      (with-out-slots (res)
        (check (ffi:mlx-custom-function res (ptr (tree-closure-closure tc))
                                        (p vjp-c) (p jvp-c) (p vmap-c))
               "custom-function")
        (let ((handle (%wrap-mlx-closure (cffi:mem-ref res :pointer))))
          (mlx:persist handle (tree-closure-closure tc) vjp-c jvp-c vmap-c)
          (tree-closure-caller tc handle))))))

(defun mlx:custom-vjp (function vjp)
  "FUNCTION with the custom vector-Jacobian product VJP; see CUSTOM-FUNCTION."
  (mlx:custom-function function :vjp vjp))

;;; Compilation control

(defun mlx:enable-compile () (check (ffi:mlx-enable-compile)) t)
(defun mlx:disable-compile () (check (ffi:mlx-disable-compile)) nil)

(defun mlx:set-compile-mode (mode)
  "MODE is :DISABLED, :NO-SIMPLIFY, :NO-FUSE or :ENABLED."
  (check (ffi:mlx-set-compile-mode mode))
  mode)

(defun mlx:clear-compile-cache ()
  (check (ffi:mlx-detail-compile-clear-cache))
  nil)
