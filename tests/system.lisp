;;;; tests/system.lisp -- devices, streams, memory, kernels, graphs, raw FFI

(in-package :mlx-tests)

(def-suite :mlx.system :in :mlx)
(in-suite :mlx.system)

(test version
  (is (stringp (mx:version)))
  (is (find #\. (mx:version))))

(test devices
  (let ((d (mx:default-device)))
    (is (typep d 'mx:mlx-device))
    (is (member (mx:device-type d) '(:cpu :gpu)))
    (is (= 0 (mx:device-index d))))
  (is (mx:device-available-p :cpu))
  (is (plusp (mx:device-count :cpu)))
  (when (mx:metal-available-p)
    (let ((info (mx:device-info :gpu)))
      (is (stringp (getf info :device-name)))
      (is (integerp (getf info :memory-size))))))

(test default-device-switching
  (let ((original (mx:device-type (mx:default-device))))
    (unwind-protect
         (progn
           (mx:set-default-device :cpu)
           (is (eq :cpu (mx:device-type (mx:default-device))))
           (is (close-to (mx:add (mx:ones '(2)) 1) '(2 2))))
      (mx:set-default-device original))))

(test streams
  (let ((s (mx:make-stream :cpu)))
    (is (typep s 'mx:mlx-stream))
    (is (eq :cpu (mx:device-type (mx:stream-device s))))
    (is (integerp (mx:stream-index s)))
    (is (close-to (mx:add (mx:ones '(2)) 1 :stream s) '(2 2)))
    (mx:with-stream (s)
      (is (close-to (mx:multiply (mx:ones '(2)) 3) '(3 3))))
    (mx:synchronize s))
  (is (close-to (mx:add 1 1 :stream :cpu) 2))
  (mx:with-device (:cpu)
    (is (close-to (mx:sum (mx:ones '(4))) 4)))
  (is (typep (mx:default-stream :cpu) 'mx:mlx-stream))
  (is (null (mx:synchronize))))

(test memory
  (mx:reset-peak-memory)
  (let ((a (mx:eval (mx:ones '(1024 1024)))))
    (is (>= (mx:active-memory) (* 4 1024 1024)))
    (is (>= (mx:peak-memory) (* 4 1024 1024)))
    (mx:free a))
  (is (integerp (mx:cache-memory)))
  (is (integerp (mx:memory-limit)))
  (let ((old (mx:set-cache-limit (* 1024 1024 1024))))
    (is (integerp old))
    (mx:set-cache-limit old))
  (is (null (mx:clear-cache))))

(test metal-kernel
  (if (not (mx:metal-available-p))
      (skip "Metal not available")
      (let ((k (fast:metal-kernel
                :name "myexp" :input-names '("inp") :output-names '("out")
                :source "uint elem = thread_position_in_grid.x;
                         T tmp = inp[elem];
                         out[elem] = metal::exp(tmp);")))
        (let ((out (funcall k :inputs (list (mx:from-lisp '(0.0 1.0 2.0)))
                              :template '(("T" . :float32))
                              :grid '(3) :threadgroup '(3)
                              :output-shapes '((3)) :output-dtypes '(:float32)
                              :stream :gpu)))            ; Metal kernels are GPU-only
          (is (= 1 (length out)))
          (is (close-to (first out) (list 1 (exp 1.0) (exp 2.0))))))))

(test graph-export
  (let* ((x (mx:ones '(2)))
         (y (mx:add (mx:multiply x 2) 1)))
    (let ((dot (mx:export-to-dot y :names (list (cons x "input")))))
      (is (search "digraph" dot))
      (is (search "input" dot)))
    (is (plusp (length (mx:print-graph y))))))

(test distributed-singleton
  (let ((group (dist:init)))
    (is (= 0 (dist:group-rank group)))
    (is (= 1 (dist:group-size group)))
    (is (close-to (dist:all-sum (mx:from-lisp '(1.0 2.0)) :group group :stream :cpu) '(1 2)))))

(test raw-ffi-layer
  ;; the generated bindings are usable directly
  (let ((a (mlx-ffi:mlx-array-new-float32 2.5)))
    (unwind-protect
         (cffi:with-foreign-object (out :float)
           (is (zerop (mlx-ffi:mlx-array-item-float32 out a)))
           (is (= 2.5 (cffi:mem-ref out :float)))
           (is (eq :float32 (mlx-ffi:mlx-array-dtype a))))
      (mlx-ffi:mlx-array-free a))))

(test gc-finalizes-arrays
  ;; arrays dropped without FREE are reclaimed by finalizers
  (dotimes (i 50) (mx:ones '(256 256)))
  (sb-ext:gc :full t)
  (sb-kernel:run-pending-finalizers)
  (is (< (mx:active-memory) (* 50 256 256 4))))

;;; ------------------------------------------------------------------
;;; ABI conventions: each by-value trick in the bindings, pinned against the
;;; live library (see the header of src/ffi/bindings.lisp)

(def-suite :mlx.abi :in :mlx)
(in-suite :mlx.abi)

(test optional-int-is-read-by-value
  ;; group_size travels as a packed mlx_optional_int
  (let ((w (random:normal :shape '(8 128))))
    (destructuring-bind (wq scales biases) (mx:quantize w :group-size 32 :bits 4)
      (declare (ignore wq biases))
      (is (equal '(8 4) (mx:shape scales))))
    (destructuring-bind (wq scales biases) (mx:quantize w :group-size 128 :bits 8)
      (declare (ignore biases))
      (is (equal '(8 1) (mx:shape scales)))
      (is (equal '(8 32) (mx:shape wq))))
    ;; has_value = false selects MLX's default (64)
    (is (equal '(8 2) (mx:shape (second (mx:quantize w)))))))

(test optional-float-is-read-by-value
  ;; default scale is 1/sqrt(n); an explicit scale must be honoured exactly
  (let ((x (mx:ones '(4))))
    (is (close-to (mx:ref (mx:hadamard-transform x) 0) 2.0))   ; H.1 = (4 0 0 0), times 1/sqrt 4
    (is (close-to (mx:ref (mx:hadamard-transform x :scale 1.0) 0) 4.0))
    (is (close-to (mx:ref (mx:hadamard-transform x :scale 0.25) 0) 1.0))))

(test optional-dtype-is-read-by-value
  (destructuring-bind (wq scales biases) (mx:quantize (random:normal :shape '(4 64)))
    (is (eq :float16 (mx:dtype (mx:dequantize wq scales :biases biases :dtype :float16))))
    (is (eq :float32 (mx:dtype (mx:dequantize wq scales :biases biases))))))

(test map-iterator-two-word-struct
  ;; the iterator's second word is passed back as the map pointer
  (let* ((entries (loop for i below 5 collect (cons (format nil "k~D" i) (mx:scalar i))))
         (map (mlx.impl::make-map-string-to-array entries)))
    (unwind-protect
         (let ((back (mlx.impl::map-string-to-array->alist map)))
           (is (= 5 (length back)))
           (is (equal (sort (mapcar #'car back) #'string<) (mapcar #'car entries)))
           (is (every (lambda (e) (= (mx:item (cdr e)) (parse-integer (car e) :start 1))) back)))
      (mlx-ffi:mlx-map-string-to-array-free map))))

(test raw-ffi-masks-float-traps
  ;; Metal's allocator leaves FP exception flags set; unmasked, SBCL would
  ;; signal FLOATING-POINT-OVERFLOW after these calls return
  (finishes (dotimes (i 2000)
              (mlx-ffi:mlx-array-free (mlx-ffi:mlx-array-new-float32 1.0)))))

;;; ------------------------------------------------------------------
;;; performance-related behaviour

(def-suite :mlx.perf :in :mlx)
(in-suite :mlx.perf)

(test scalar-constants-are-cached
  (let ((a (mx:ones '(3))))
    (mx:with-scope () (mx:add a 0.125))
    ;; the cached scalar survives the scope and is reused correctly
    (is (close-to (mx:add a 0.125) '(1.125 1.125 1.125)))
    (is (close-to (mx:multiply (mx:ones '(2) :dtype :float16) 0.125) '(0.125 0.125)))
    (is (eq :float16 (mx:dtype (mx:multiply (mx:ones '(2) :dtype :float16) 0.125))))
    (is (eq :float32 (mx:dtype (mx:multiply a 0.125))))))

(test scoped-handles-finalized-only-when-escaping
  (let (inner outer)
    (setf outer (mx:with-scope ()
                  (setf inner (mx:ones '(2)))
                  (is (not (mlx.impl::handle-finalized inner)) "no finalizer inside a scope")
                  (mx:zeros '(2))))
    (is (mx:freed-p inner))
    (is (mlx.impl::handle-finalized outer) "escaping the outermost scope attaches one"))
  (is (mlx.impl::handle-finalized (mx:ones '(1))) "outside any scope, finalized at once"))
