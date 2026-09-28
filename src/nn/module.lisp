;;;; nn/module.lisp -- the module protocol (mlx.nn.Module)
;;;;
;;;; A module is a funcallable CLOS instance: (funcall model x) calls
;;;; FORWARD.  Its parameters and submodules are named children, kept in
;;;; insertion order, whose values are arrays, modules, or lists of them.
;;;; Child names are strings following the Python/Hugging Face convention
;;;; ("q_proj"), so weight files load by name; a symbol such as :Q-PROJ is
;;;; accepted anywhere a name is and means "q_proj".
;;;;
;;;; PARAMETERS returns the arrays as a tree of hash tables (string keys)
;;;; and lists, the shape GRAD and the optimizers work over.

(in-package :mlx.nn.impl)

(defclass nn:module (sb-mop:funcallable-standard-object)
  ((children :initform '() :accessor %children
             :documentation "Alist of (name . value), in insertion order.")
   (frozen :initform '() :accessor %frozen
           :documentation "Names of this module's array children excluded from training.")
   (training :initform t :accessor nn:training-p))
  (:metaclass sb-mop:funcallable-standard-class)
  (:documentation "Base class of neural network modules."))

(defmacro nn:defmodule (name direct-superclasses slots &rest options)
  "DEFCLASS for a module: adds NN:MODULE as a superclass if needed and the
funcallable metaclass modules require."
  `(defclass ,name ,(if (some (lambda (c) (subtypep c 'nn:module)) direct-superclasses)
                        direct-superclasses
                        (append direct-superclasses '(nn:module)))
     ,slots
     (:metaclass sb-mop:funcallable-standard-class)
     ,@options))

(defgeneric nn:forward (module &rest args)
  (:documentation "Compute MODULE's output.  Called by (FUNCALL module args...)."))

(defmethod initialize-instance :after ((m nn:module) &key)
  (sb-mop:set-funcallable-instance-function
   m (lambda (&rest args) (apply #'nn:forward m args))))

(defun name-string (name)
  (etypecase name
    (string name)
    (symbol (substitute #\_ #\- (string-downcase (symbol-name name))))
    (integer (princ-to-string name))))

;;; ------------------------------------------------------------------
;;; Children

(defun nn:children (module)
  "MODULE's children as an alist of (name . value), in insertion order."
  (copy-alist (%children module)))

(defun nn:child (module name)
  "The child NAME (string or symbol) of MODULE, or NIL."
  (cdr (assoc (name-string name) (%children module) :test #'string=)))

(defvar *persist-children* t
  "When true, arrays stored in a module are exempted from WITH-SCOPE (the
module owns them).  NN:VALUE-AND-GRAD binds it to NIL for its temporary
swap of tracer arrays into the model.")

(defun own (value)
  "Persist the arrays in VALUE (an array or list) if modules own them."
  (when *persist-children*
    (typecase value
      (mx:mlx-array (mx:persist value))
      (cons (dolist (v value) (when (typep v 'mx:mlx-array) (mx:persist v))))))
  value)

(defun (setf nn:child) (value module name)
  (own value)
  (let* ((name (name-string name))
         (cell (assoc name (%children module) :test #'string=)))
    (if cell
        (setf (cdr cell) value)
        (setf (%children module) (append (%children module) (list (cons name value)))))
    value))

(defun nn:register (module name value)
  "Add or replace MODULE's child NAME.  Returns VALUE."
  (setf (nn:child module name) value))

(defun nn:modules (module)
  "MODULE and all its descendant modules, depth first."
  (let ((out '()))
    (labels ((walk (x)
               (typecase x
                 (nn:module (push x out)
                  (loop for (nil . v) in (%children x) do (walk v)))
                 (cons (mapc #'walk x)))))
      (walk module))
    (nreverse out)))

(defun nn:apply-to-modules (function module)
  "Call FUNCTION with (path module) for MODULE and every descendant; PATH
is the dotted name (\"\" for MODULE itself)."
  (labels ((walk (x path)
             (typecase x
               (nn:module (funcall function path x)
                (loop for (name . v) in (%children x)
                      do (walk v (if (string= path "") name (format nil "~A.~A" path name)))))
               (cons (loop for v in x for i from 0
                           do (walk v (format nil "~A.~D" path i)))))))
    (walk module ""))
  module)

;;; ------------------------------------------------------------------
;;; Parameters

(defun frozen-p (module name) (member name (%frozen module) :test #'string=))

(defun parameter-subtree (value trainable-only)
  (typecase value
    (mx:mlx-array value)
    (nn:module (let ((table (collect-parameters value trainable-only)))
                 (and (plusp (hash-table-count table)) table)))
    (cons (let ((subs (mapcar (lambda (v) (parameter-subtree v trainable-only)) value)))
            (when (some #'identity subs) subs)))
    (t nil)))

(defun collect-parameters (module trainable-only)
  (let ((table (make-hash-table :test 'equal)))
    (loop for (name . value) in (%children module)
          unless (and trainable-only (typep value 'mx:mlx-array) (frozen-p module name))
            do (let ((sub (parameter-subtree value trainable-only)))
                 (when (and sub (not (and (hash-table-p sub) (zerop (hash-table-count sub)))))
                   (setf (gethash name table) sub))))
    table))

(defun nn:parameters (module)
  "All of MODULE's arrays as a tree: hash tables keyed by child name, and
lists for list children (NIL where an element has none)."
  (collect-parameters module nil))

(defun nn:trainable-parameters (module)
  "Like PARAMETERS, without frozen arrays."
  (collect-parameters module t))

(defun nn:flatten-parameters (tree)
  "A parameter TREE (or a module) as an alist of (\"dotted.name\" . array)."
  (let ((tree (if (typep tree 'nn:module) (nn:parameters tree) tree))
        (out '()))
    (impl::map-leaves-with-path
     (lambda (path leaf)
       (push (cons (format nil "~{~A~^.~}" path) leaf) out))
     tree)
    (nreverse out)))

(defun nn:parameter-count (module)
  (loop for (nil . a) in (nn:flatten-parameters module) sum (mx:size a)))

(defun nn:update (module tree)
  "Replace MODULE's arrays with those in TREE (shaped like PARAMETERS;
missing entries are left alone).  Returns MODULE."
  (labels ((update-value (old new)
             ;; returns the value to store
             (cond ((null new) old)
                   ((typep old 'nn:module) (update-module old new) old)
                   ((and (listp old) (listp new) old)
                    (loop for o in old
                          for rest = new then (cdr rest)
                          collect (own (update-value o (car rest)))))
                   (t new)))
           (update-module (m table)
             (etypecase table
               (hash-table
                (maphash (lambda (name new)
                           (let ((cell (assoc name (%children m) :test #'string=)))
                             (if cell
                                 (setf (cdr cell) (own (update-value (cdr cell) new)))
                                 (nn:register m name new))))
                         table)))))
    (update-module module tree)
    module))

(defun nn:freeze (module &key keys (recurse t))
  "Exclude MODULE's arrays (only those named in KEYS, if given) from
TRAINABLE-PARAMETERS; with RECURSE, also those of its submodules.
Returns MODULE."
  (dolist (m (if recurse (nn:modules module) (list module)))
    (loop for (name . value) in (%children m)
          when (and (typep value 'mx:mlx-array)
                    (or (null keys) (member name (mapcar #'name-string keys) :test #'string=)))
            do (pushnew name (%frozen m) :test #'string=)))
  module)

(defun nn:unfreeze (module &key keys (recurse t))
  "Undo FREEZE (for KEYS, if given).  Returns MODULE."
  (dolist (m (if recurse (nn:modules module) (list module)))
    (setf (%frozen m)
          (if keys
              (set-difference (%frozen m) (mapcar #'name-string keys) :test #'string=)
              '())))
  module)

(defun nn:train-mode (module &optional (on t))
  "Put MODULE and its submodules in training mode (ON true) or evaluation
mode (ON false), which e.g. disables dropout.  Returns MODULE."
  (dolist (m (nn:modules module)) (setf (nn:training-p m) (and on t)))
  module)

;;; ------------------------------------------------------------------
;;; Weights

(defun weight-source-alist (source)
  (etypecase source
    ((or string pathname)
     (let ((path (pathname source)))
       (if (uiop:directory-pathname-p path)
           (loop for file in (directory (merge-pathnames "*.safetensors" path))
                 nconc (weight-source-alist file))
           (let ((table (mx:load path)))
             (unless (hash-table-p table)
               (error "~A does not contain named weights." path))
             (sort (loop for k being the hash-keys of table using (hash-value v) collect (cons k v))
                   #'string< :key #'car)))))
    (hash-table (loop for k being the hash-keys of source using (hash-value v) collect (cons k v)))
    (list source)))

(defun locate-slot (module parts)
  "Resolve dotted-name PARTS below MODULE.  Returns (values container key)
where CONTAINER is the module (key: child name) or list cons (key: NIL)
holding the final value, or NIL if the path does not exist."
  (let ((obj module))
    (loop for (part . more) on parts
          do (etypecase obj
               (nn:module
                (let ((cell (assoc part (%children obj) :test #'string=)))
                  (cond ((null more) (return-from locate-slot (values obj part)))
                        ((null cell) (return-from locate-slot nil))
                        (t (setf obj (cdr cell))))))
               (list
                (let* ((i (ignore-errors (parse-integer part)))
                       (cell (and i (nthcdr i obj))))
                  (cond ((null cell) (return-from locate-slot nil))
                        ((null more) (return-from locate-slot (values cell nil)))
                        (t (setf obj (car cell))))))
               (t (return-from locate-slot nil))))))

(defun nn:load-weights (module source &key (strict t))
  "Load arrays into MODULE by dotted name (\"layers.0.weight\").  SOURCE is
a .safetensors/.npz file, a directory of .safetensors files, a hash table
or an alist.  With STRICT, unknown names, missing parameters and shape
mismatches are errors; otherwise unknown names are skipped.  Returns MODULE."
  (let ((seen (make-hash-table :test 'equal))
        (unknown '()))
    (loop for (name . value) in (weight-source-alist source)
          for array = (mx:ensure-array value)
          do (multiple-value-bind (container key)
                 (locate-slot module (uiop:split-string name :separator "."))
               (let ((old (cond ((null container) nil)
                                ((typep container 'nn:module) (nn:child container key))
                                (t (car container)))))
                 (cond ((null container) (push name unknown))
                       ((and strict (not (typep old 'mx:mlx-array)))
                        (push name unknown))
                       (t
                        (when (and strict (not (equal (mx:shape old) (mx:shape array))))
                          (error "Weight ~A has shape ~A; the model expects ~A."
                                 name (mx:shape array) (mx:shape old)))
                        (setf (gethash name seen) t)
                        (if (typep container 'nn:module)
                            (setf (nn:child container key) array)
                            (setf (car container) (own array))))))))
    (when strict
      (let ((missing (loop for (name . nil) in (nn:flatten-parameters module)
                           unless (gethash name seen) collect name)))
        (when (or unknown missing)
          (error "Weights do not match the model.~@[~%Unknown: ~{~A~^, ~}~]~@[~%Missing: ~{~A~^, ~}~]"
                 (subseq-at-most (reverse unknown) 20) (subseq-at-most missing 20)))))
    module))

(defun subseq-at-most (list n)
  (if (> (length list) n) (append (subseq list 0 n) (list "...")) list))

(defun nn:save-weights (module file)
  "Save MODULE's parameters to FILE (.safetensors) under dotted names."
  (mx:save-safetensors file (nn:flatten-parameters module))
  file)

;;; ------------------------------------------------------------------
;;; Training

(defun nn:value-and-grad (model function)
  "Return a function of ARGS computing (values (FUNCTION args...) gradients),
the gradients being with respect to MODEL's trainable parameters (a tree
shaped like TRAINABLE-PARAMETERS).  FUNCTION calls MODEL itself."
  (let ((vg (mx:value-and-grad (lambda (params &rest args)
                                 (let ((*persist-children* nil))
                                   (nn:update model params))
                                 (apply function args)))))
    (lambda (&rest args)
      (let ((params (nn:trainable-parameters model)))
        (unwind-protect (apply vg params args)
          ;; tracing left tracer arrays in the model; put the real ones back
          (nn:update model params))))))

;;; ------------------------------------------------------------------
;;; Printing

(defgeneric module-description (module)
  (:documentation "Short text describing MODULE's configuration, or NIL.")
  (:method ((m nn:module)) nil))

(defmethod print-object ((m nn:module) stream)
  (print-unreadable-object (m stream :type t :identity (null (module-description m)))
    (let ((d (module-description m)))
      (when d (write-string d stream)))))

(defun nn:summary (module &optional (stream *standard-output*))
  "Print MODULE's tree of submodules and parameter counts."
  (labels ((walk (name x depth)
             (typecase x
               (nn:module
                (format stream "~&~vT~@[~A: ~]~A~@[ ~A~]~%" (* 2 depth) name
                        (string-downcase (class-name (class-of x))) (module-description x))
                (loop for (n . v) in (%children x) do (walk n v (1+ depth))))
               (mx:mlx-array
                (format stream "~&~vT~A ~(~A~) ~A~%" (* 2 depth) name (mx:dtype x) (mx:shape x)))
               (cons (loop for v in x for i from 0
                           do (walk (format nil "~A.~D" name i) v depth))))))
    (walk nil module 0)
    (format stream "~&~:D parameters~%" (nn:parameter-count module))
    (values)))
