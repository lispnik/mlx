;;;; tree.lisp -- "pytrees": nested Lisp structures holding arrays
;;;;
;;;; Transformations such as GRAD accept arguments that are arrays or trees
;;;; of arrays: conses (lists, alists, plists), simple vectors and hash
;;;; tables.  Leaves are MLX arrays and Lisp numbers; everything else
;;;; (keywords, strings, NIL...) is structure and passes through unchanged.

(in-package :mlx.impl)

(defstruct (leaf-slot (:constructor make-leaf-slot ())) "Placeholder for a leaf.")
(defstruct (hash-node (:constructor make-hash-node (test entries))) test entries)

(defun leafp (x) (typep x '(or mlx:mlx-array number)))

(defun mlx:tree-flatten (tree)
  "Returns (values leaves structure): the list of leaves in depth-first
order, and a skeleton from which TREE-UNFLATTEN rebuilds the tree."
  (let ((leaves '()))
    (labels ((walk (x)
               (cond ((leafp x) (push x leaves) (make-leaf-slot))
                     ((consp x) (cons (walk (car x)) (walk (cdr x))))
                     ((and (simple-vector-p x)) (map 'simple-vector #'walk x))
                     ((hash-table-p x)
                      (let ((entries '()))
                        (maphash (lambda (k v) (push (cons k v) entries)) x)
                        (make-hash-node (hash-table-test x)
                                        (mapcar (lambda (e) (cons (car e) (walk (cdr e))))
                                                (nreverse entries)))))
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
