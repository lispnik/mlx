;;;; tree.lisp -- "pytrees": nested Lisp structures holding arrays
;;;;
;;;; Transformations such as GRAD accept arguments that are arrays or trees
;;;; of arrays: conses (lists, alists, plists), simple vectors and hash
;;;; tables.  Leaves are MLX arrays and Lisp numbers; everything else
;;;; (keywords, strings, NIL...) is structure and passes through unchanged.
;;;;
;;;; Hash-table entries are visited in sorted key order, so two tables with
;;;; the same keys flatten identically however they were built -- TREE-MAP
;;;; over parameters and gradients depends on it.

(in-package :mlx.impl)

(defstruct (leaf-slot (:constructor make-leaf-slot ())) "Placeholder for a leaf.")
(defstruct (hash-node (:constructor make-hash-node (test entries))) test entries)

(defun leafp (x) (typep x '(or mlx:mlx-array number)))

(defun key-order (a b)
  "Deterministic order for hash-table keys: numbers numerically, then
everything else by printed representation."
  (cond ((and (realp a) (realp b)) (< a b))
        ((realp a) t)
        ((realp b) nil)
        (t (string< (princ-to-string a) (princ-to-string b)))))

(defun sorted-hash-entries (table)
  (let ((entries '()))
    (maphash (lambda (k v) (push (cons k v) entries)) table)
    (sort entries #'key-order :key #'car)))

(defun mlx:tree-flatten (tree)
  "Returns (values leaves structure): the list of leaves in depth-first
order, and a skeleton from which TREE-UNFLATTEN rebuilds the tree."
  (let ((leaves '()))
    (labels ((walk (x)
               (cond ((leafp x) (push x leaves) (make-leaf-slot))
                     ((consp x) (cons (walk (car x)) (walk (cdr x))))
                     ((and (simple-vector-p x)) (map 'simple-vector #'walk x))
                     ((hash-table-p x)
                      (make-hash-node (hash-table-test x)
                                      (mapcar (lambda (e) (cons (car e) (walk (cdr e))))
                                              (sorted-hash-entries x))))
                     (t x))))
      (let ((structure (walk tree)))
        (values (nreverse leaves) structure)))))

(defun mlx:tree-unflatten (structure leaves)
  "Rebuild a tree from STRUCTURE (from TREE-FLATTEN) and a list of LEAVES."
  (labels ((walk (x)
             (cond ((leaf-slot-p x)
                    (when (null leaves) (error "Not enough leaves to fill the tree."))
                    (pop leaves))
                   ((consp x) (let ((a (walk (car x)))) (cons a (walk (cdr x)))))
                   ((simple-vector-p x) (map 'simple-vector #'walk x))
                   ((hash-node-p x)
                    (let ((h (make-hash-table :test (hash-node-test x))))
                      (loop for (k . v) in (hash-node-entries x) do (setf (gethash k h) (walk v)))
                      h))
                   (t x))))
    (walk structure)))

(defun mlx:tree-map (function tree &rest more-trees)
  "Apply FUNCTION to corresponding leaves of TREE and MORE-TREES (which must
share TREE's structure); returns a tree shaped like TREE."
  (multiple-value-bind (leaves structure) (mlx:tree-flatten tree)
    (let ((others (mapcar (lambda (tr) (mlx:tree-flatten tr)) more-trees)))
      (mlx:tree-unflatten structure (apply #'mapcar function leaves others)))))

(defun leaf-count (tree) (length (mlx:tree-flatten tree)))

(defun map-leaves-with-path (function tree)
  "Call FUNCTION with (path leaf) for each leaf of TREE, in TREE-FLATTEN
order.  PATH is a list of hash-table keys and list/vector indices from the
root; for a plist the key symbol stands in for the value's position."
  (labels ((walk (x path)
             (cond ((leafp x) (funcall function (reverse path) x))
                   ((hash-table-p x)
                    (loop for (k . v) in (sorted-hash-entries x) do (walk v (cons k path))))
                   ((simple-vector-p x)
                    (loop for v across x for i from 0 do (walk v (cons i path))))
                   ((consp x)
                    (loop for rest on x for i from 0
                          do (walk (car rest) (cons i path))
                          unless (listp (cdr rest)) do (walk (cdr rest) (cons :cdr path)) (return))))))
    (walk tree '())
    nil))
