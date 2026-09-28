;;;; ops-extra.lisp -- operations with a friendlier interface than the C API:
;;;; Python-style argument handling, and NumPy-style indexing with REF.

(in-package :mlx.impl)

(defun mlx:arange (&rest args)
  "(arange stop), (arange start stop) or (arange start stop step), plus
keyword arguments :DTYPE and :STREAM.  DTYPE defaults to :INT32 when all
bounds are integers, else :FLOAT32."
  (let* ((kpos (position-if #'keywordp args))
         (nums (subseq args 0 kpos))
         (keys (and kpos (subseq args kpos))))
    (destructuring-bind (&key dtype stream) keys
      (destructuring-bind (start stop step)
          (ecase (length nums)
            (1 (list 0 (first nums) 1))
            (2 (list (first nums) (second nums) 1))
            (3 nums))
        (%arange (float start 1d0) (float stop 1d0) (float step 1d0)
                 :dtype (or dtype (if (every #'integerp nums) :int32 :float32))
                 :stream stream)))))

(defun mlx:pad (a pad-width &key (mode "constant") (constant-values 0) stream)
  "Pad A.  PAD-WIDTH is an integer (same before/after on every axis), a
pair (before after) for every axis, or a list of pairs, one per axis.
MODE is \"constant\" or \"edge\"."
  (let ((mode (string-downcase (string mode))))
    (if (integerp pad-width)
        (%pad-symmetric a pad-width :pad-value constant-values :mode mode :stream stream)
        (let* ((n (mlx:ndim a))
               (pairs (if (every #'integerp pad-width)
                          (make-list n :initial-element pad-width)
                          pad-width)))
          (unless (= (length pairs) n)
            (error "PAD-WIDTH ~S does not match rank ~D." pad-width n))
          (%pad a (all-axes a) (mapcar #'first pairs) (mapcar #'second pairs)
                :pad-value constant-values :mode mode :stream stream)))))

(defun mlx:split (a indices-or-sections &key (axis 0) stream)
  "Split A along AXIS into INDICES-OR-SECTIONS equal parts (an integer) or
at the given indices (a list).  Returns a list of arrays."
  (if (integerp indices-or-sections)
      (%split a indices-or-sections axis :stream stream)
      (%split-sections a indices-or-sections axis :stream stream)))

(defun mlx:tensordot (a b &key (axes 2) stream)
  "Tensor contraction.  AXES is an integer (contract the last AXES axes of
A with the first of B) or a list of two axis lists."
  (if (integerp axes)
      (%tensordot-axis a b axes :stream stream)
      (%tensordot a b (first axes) (second axes) :stream stream)))

(defun mlx.linalg:norm (a &key ord axis keepdims stream)
  "Matrix or vector norm.  ORD: NIL (2-norm / Frobenius), a number, :INF,
:-INF, or a string such as \"fro\" or \"nuc\".  AXIS: NIL, an integer or a list."
  (let ((axis (if (integerp axis) (list axis) axis)))
    (etypecase ord
      (null (mlx.linalg:norm-l2 a :axis axis :keepdims keepdims :stream stream))
      (real (mlx.linalg:norm-ord a ord :axis axis :keepdims keepdims :stream stream))
      ((member :inf :-inf)
       (mlx.linalg:norm-ord a (if (eq ord :inf)
                                  sb-ext:double-float-positive-infinity
                                  sb-ext:double-float-negative-infinity)
                            :axis axis :keepdims keepdims :stream stream))
      ((or string symbol)
       (mlx.linalg:norm-matrix a (string-downcase (string ord))
                               :axis axis :keepdims keepdims :stream stream)))))

;;; random

(defun mlx.random:seed (seed)
  "Seed the global random generator."
  (check (ffi:mlx-random-seed seed) "random seed")
  seed)

(defun mlx.random:split (key &key (num 2) stream)
  "Split the PRNG KEY into NUM keys, returned as an array of shape (NUM 2)."
  (%split-num key :num num :stream stream))

(defun mlx.random:categorical (logits &key (axis -1) shape num-samples key stream)
  "Sample category indices from the unnormalized LOGITS along AXIS.  Give
at most one of SHAPE (output shape) and NUM-SAMPLES."
  (cond ((and shape num-samples)
         (error "Give only one of :SHAPE and :NUM-SAMPLES."))
        (shape (%categorical-shape logits shape :axis axis :key key :stream stream))
        (num-samples (%categorical-num-samples logits num-samples :axis axis :key key
                                                                  :stream stream))
        (t (%categorical logits :axis axis :key key :stream stream))))

(defun mlx.random:permutation (x &key (axis 0) key stream)
  "A random permutation of (arange X) if X is an integer, else of X's
entries along AXIS."
  (if (integerp x)
      (%permutation-arange x :key key :stream stream)
      (%permutation x :axis axis :key key :stream stream)))

;;; ------------------------------------------------------------------
;;; Indexing
;;;
;;; (ref a spec...) mirrors a[spec, ...] in NumPy/MLX.  Each spec is:
;;;   integer          select one index; the axis is dropped
;;;   T or :all        the whole axis
;;;   (start stop [step])  a slice; any bound may be NIL (a list is always a slice)
;;;   :newaxis         insert an axis of length 1
;;;   an MLX-ARRAY or a vector of integers, e.g. #(2 0)   gather along the axis
;;; Missing trailing specs mean the whole axis.

(defun normalize-slice (spec n)
  "START, STOP, STEP for the slice SPEC on an axis of length N."
  (destructuring-bind (&optional start stop (step 1)) spec
    (let ((step (or step 1)))
      (when (zerop step) (error "Slice step cannot be zero."))
      (flet ((clamp (i lo hi) (max lo (min hi i)))
             (wrap (i) (if (minusp i) (+ i n) i)))
        (if (plusp step)
            (values (if start (clamp (wrap start) 0 n) 0)
                    (if stop (clamp (wrap stop) 0 n) n)
                    step)
            (values (if start (clamp (wrap start) -1 (1- n)) (1- n))
                    (if stop (clamp (wrap stop) -1 (1- n)) -1)
                    step))))))

(defun slice-length (start stop step)
  (if (plusp step)
      (max 0 (ceiling (- stop start) step))
      (max 0 (ceiling (- start stop) (- step)))))

(defun parse-index (a specs)
  "Plan for indexing A with SPECS.  Returns a plist with :STARTS :STOPS
:STRIDES (for one slice over all axes), :DROP (axes to squeeze after the
slice), :NEWAXES (positions to insert, in output coordinates), :GATHERS
((axis . indices) in post-slice coordinates) and :SLICE-SHAPE."
  (let* ((shape (mlx:shape a))
         (n (length shape))
         (consumed (count-if-not (lambda (s) (eq s :newaxis)) specs)))
    (when (> consumed n)
      (error "Too many indices (~D) for array of rank ~D." consumed n))
    (let ((starts '()) (stops '()) (strides '()) (drop '()) (newaxes '()) (gathers '())
          (slice-shape '()) (axis 0) (out-axis 0))
      (dolist (spec (append specs (make-list (- n consumed) :initial-element t)))
        (if (eq spec :newaxis)
            (progn (push out-axis newaxes) (incf out-axis))
            (let ((len (nth axis shape)))
              (etypecase spec
                (integer
                 (let ((i (if (minusp spec) (+ spec len) spec)))
                   (unless (< -1 i len)
                     (error "Index ~D out of bounds for axis ~D of size ~D." spec axis len))
                   (push i starts) (push (1+ i) stops) (push 1 strides)
                   (push 1 slice-shape) (push axis drop)))
                ((or (eql t) (eql :all))
                 (push 0 starts) (push len stops) (push 1 strides)
                 (push len slice-shape) (incf out-axis))
                ((or mlx:mlx-array (and vector (not string)))
                 (push 0 starts) (push len stops) (push 1 strides)
                 (push len slice-shape)
                 (push (cons axis spec) gathers) (incf out-axis))
                (list
                 (multiple-value-bind (b e s) (normalize-slice spec len)
                   ;; -1 means "before index 0" here, but C++ would wrap it
                   (push (if (minusp b) (- b len) b) starts)
                   (push (if (minusp e) (- e len) e) stops)
                   (push s strides)
                   (push (slice-length b e s) slice-shape) (incf out-axis))))
              (incf axis))))
      (list :starts (nreverse starts) :stops (nreverse stops) :strides (nreverse strides)
            :drop (nreverse drop) :newaxes (nreverse newaxes) :gathers (nreverse gathers)
            :slice-shape (nreverse slice-shape)))))

(defun trivial-slice-p (plan shape)
  (and (every #'zerop (getf plan :starts))
       (equal (getf plan :stops) shape)
       (every (lambda (s) (= s 1)) (getf plan :strides))))

(defun mlx:ref (a &rest specs)
  "NumPy-style indexing: (ref a 0 '(1 nil) t :newaxis) is a[0, 1:, :, None].
See the INDEXING section of ops-extra.lisp for the spec forms."
  (let* ((a (mlx:ensure-array a))
         (plan (parse-index a specs))
         (result (if (trivial-slice-p plan (mlx:shape a))
                     a
                     (mlx:slice a (getf plan :starts) (getf plan :stops)
                                :strides (getf plan :strides)))))
    ;; gathers happen before dropping axes so their axis numbers hold
    (loop for (axis . indices) in (getf plan :gathers)
          do (setf result (mlx:take-axis result (if (typep indices 'mlx:mlx-array)
                                                    indices
                                                    (mlx:from-lisp indices :dtype :int32))
                                         axis)))
    (when (getf plan :drop)
      (setf result (mlx:squeeze-axes result (getf plan :drop))))
    (when (getf plan :newaxes)
      (setf result (mlx:expand-dims-axes result (getf plan :newaxes))))
    result))

(defun (setf mlx:ref) (value a &rest specs)
  "Assign VALUE (an array or scalar, broadcast as needed) into the region of
A selected by SPECS (integers, T and slices; no gathers or :NEWAXIS).  A is
updated in place -- its handle now refers to the updated array."
  (let* ((plan (parse-index a specs)))
    (when (or (getf plan :gathers) (getf plan :newaxes))
      (error "(SETF REF) supports integers, T and slices only."))
    (let* ((slice-shape (getf plan :slice-shape))
           (value (mlx:ensure-array value :dtype (weak-like-scalar-dtype value a)))
           (value (if (and (getf plan :drop) (plusp (mlx:ndim value))
                           (< (mlx:ndim value) (length slice-shape)))
                      ;; re-insert the dropped axes so VALUE broadcasts correctly
                      (mlx:expand-dims-axes value (getf plan :drop))
                      value))
           (updated (mlx:slice-update a value (getf plan :starts) (getf plan :stops)
                                      :strides (getf plan :strides))))
      (replace-array-contents a updated)
      value)))

(defun weak-like-scalar-dtype (value a)
  (if (typep value '(or number (eql t))) (weak-scalar-dtype value (mlx:dtype a)) nil))

(defun replace-array-contents (target source)
  "Make the handle TARGET refer to SOURCE's array; SOURCE is consumed."
  (let* ((box (handle-box target))
         (old (car box)))
    (setf (car box) (steal-pointer source))
    (when old (ffi:mlx-array-free old))
    target))
