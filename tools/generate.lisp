;;;; tools/generate.lisp -- generate CFFI bindings and high-level op specs
;;;; from the installed mlx-c headers.
;;;;
;;;; Usage (from the project root):
;;;;
;;;;   sbcl --script tools/generate.lisp [--check] [/path/to/include/mlx/c]
;;;;
;;;; With --check nothing is written: the exit status is 1 if the committed
;;;; generated files differ from what the installed headers produce.
;;;;
;;;; The include directory defaults to $MLX_C_INCLUDE, then
;;;; /opt/homebrew/include/mlx/c, then /usr/local/include/mlx/c.
;;;;
;;;; Outputs:
;;;;   src/ffi/bindings.lisp     every C function as a CFFI:DEFCFUN, plus enums
;;;;                             and handle types (package MLX-FFI)
;;;;   src/generated/ops.lisp    one DEFINE-OP form per array operation, plus
;;;;                             axis-family dispatchers (package MLX.IMPL)
;;;;   src/generated/exports.sexp  exported symbol names per public package
;;;;
;;;; This script depends on nothing but ANSI CL so it runs under --script.

(require :sb-posix)

(defpackage :mlx-generate
  (:use :cl))

(in-package :mlx-generate)

;;; ------------------------------------------------------------------
;;; Small utilities

(defun read-file (path)
  (with-open-file (in path :external-format :utf-8)
    (let ((s (make-string (file-length in))))
      (subseq s 0 (read-sequence s in)))))

(defun whitespacep (c) (member c '(#\Space #\Tab #\Newline #\Return #\Page)))

(defun trim (s) (string-trim '(#\Space #\Tab #\Newline #\Return #\Page) s))

(defun collapse-ws (s)
  (with-output-to-string (out)
    (let ((prev-ws nil))
      (loop for c across s
            do (if (whitespacep c)
                   (unless prev-ws (write-char #\Space out) (setf prev-ws t))
                   (progn (write-char c out) (setf prev-ws nil)))))))

(defun starts-with (prefix s)
  (and (>= (length s) (length prefix)) (string= prefix s :end2 (length prefix))))

(defun ends-with (suffix s)
  (and (>= (length s) (length suffix))
       (string= suffix s :start2 (- (length s) (length suffix)))))

(defun split-string (s char)
  (loop with start = 0
        for pos = (position char s :start start)
        collect (subseq s start pos)
        while pos do (setf start (1+ pos))))

(defun identifier-char-p (c) (or (alphanumericp c) (char= c #\_)))

(defun lispify (c-name)
  "mlx_array_new_data -> mlx-array-new-data; leading underscore -> %."
  (let ((s (string-downcase (substitute #\- #\_ c-name))))
    (if (starts-with "-" s)
        (concatenate 'string "%" (string-left-trim "-" s))
        s)))

;;; ------------------------------------------------------------------
;;; Lexing: strip comments and preprocessor lines, keep doc comments.

(defvar *docs* (make-array 0 :adjustable t :fill-pointer t))

(defun clean-doc (text)
  "Turn the inside of a /** ... */ comment into plain text."
  (let ((lines (mapcar (lambda (l) (trim (string-left-trim " *" (trim l))))
                       (split-string text #\Newline))))
    (setf lines (remove-if (lambda (l) (or (string= l "")
                                           (starts-with "@{" l) (starts-with "@}" l)
                                           (starts-with "\\defgroup" l)))
                           lines))
    (and lines (trim (format nil "~{~A~^ ~}" lines)))))

(defun strip-comments (text)
  "Remove comments. `/* may be null */` becomes the token __NULLABLE__;
doc comments become @@DOC<n>@@ markers indexing *DOCS*."
  (with-output-to-string (out)
    (let ((i 0) (n (length text)))
      (loop while (< i n)
            do (cond
                 ((and (< (1+ i) n) (char= (char text i) #\/) (char= (char text (1+ i)) #\*))
                  (let* ((end (search "*/" text :start2 (+ i 2)))
                         (body (subseq text (+ i 2) end))
                         (docp (and (plusp (length body)) (char= (char body 0) #\*))))
                    (cond ((string= (trim body) "may be null")
                           (write-string " __NULLABLE__ " out))
                          (docp
                           (let ((d (clean-doc (subseq body 1))))
                             (when d
                               (vector-push-extend d *docs*)
                               (format out " @@DOC~D@@ " (1- (length *docs*))))))
                          (t (write-char #\Space out)))
                    (setf i (+ end 2))))
                 ((and (< (1+ i) n) (char= (char text i) #\/) (char= (char text (1+ i)) #\/))
                  (setf i (or (position #\Newline text :start i) n)))
                 (t (write-char (char text i) out) (incf i)))))))

(defun strip-preprocessor (text)
  (with-output-to-string (out)
    (dolist (line (split-string text #\Newline))
      (let ((tl (trim line)))
        (unless (or (starts-with "#" tl)
                    (string= tl "}")
                    (search "extern \"C\"" tl))
          (write-line line out))))))

(defun split-statements (text)
  "Split on semicolons that are not nested inside () or {}."
  (let ((depth 0) (start 0) (out '()))
    (loop for i from 0 below (length text)
          for c = (char text i)
          do (case c
               ((#\( #\{) (incf depth))
               ((#\) #\}) (decf depth))
               (#\; (when (zerop depth)
                      (push (collapse-ws (trim (subseq text start i))) out)
                      (setf start (1+ i))))))
    (remove "" (nreverse out) :test #'string=)))

(defun take-docs (stmt)
  "Split leading @@DOCn@@ markers off STMT.  Returns (values stmt doc)."
  (let ((doc nil))
    (loop
      (setf stmt (trim stmt))
      (if (starts-with "@@DOC" stmt)
          (let* ((end (search "@@" stmt :start2 5)))
            (setf doc (aref *docs* (parse-integer stmt :start 5 :end end))
                  stmt (subseq stmt (+ end 2))))
          (return)))
    ;; drop any stray markers inside
    (loop for p = (search "@@DOC" stmt)
          while p
          do (let ((e (search "@@" stmt :start2 (+ p 5))))
               (setf stmt (concatenate 'string (subseq stmt 0 p) (subseq stmt (+ e 2))))))
    (values (collapse-ws (trim stmt)) doc)))

;;; ------------------------------------------------------------------
;;; Parsing declarations

(defstruct param name base (ptr 0) const nullable funcptr)
(defstruct cfun name ret params doc header proto)

(defvar *handles* '())      ; opaque {void* ctx} struct typedef names
(defvar *iterators* '())    ; {void* ctx; void* map_ctx} typedef names
(defvar *enums* '())        ; (name . ((const . value) ...))
(defvar *structs* '())      ; (name . field-strings) for other structs
(defvar *funcptr-types* '())
(defvar *functions* '())

(defun split-top-commas (s)
  (let ((depth 0) (start 0) (out '()))
    (loop for i from 0 below (length s)
          for c = (char s i)
          do (case c
               (#\( (incf depth))
               (#\) (decf depth))
               (#\, (when (zerop depth)
                      (push (trim (subseq s start i)) out)
                      (setf start (1+ i))))))
    (push (trim (subseq s start)) out)
    (remove "" (nreverse out) :test #'string=)))

(defun tokenize-type (s)
  "Split a C type/param string into identifier tokens and '*' tokens."
  (let ((tokens '()) (cur '()))
    (flet ((flush () (when cur (push (coerce (nreverse cur) 'string) tokens) (setf cur '()))))
      (loop for c across s
            do (cond ((identifier-char-p c) (push c cur))
                     ((char= c #\*) (flush) (push "*" tokens))
                     (t (flush))))
      (flush))
    (nreverse tokens)))

(defun parse-param (s)
  (let ((nullable (search "__NULLABLE__" s)))
    (when nullable
      (setf s (trim (concatenate 'string (subseq s 0 nullable)
                                 (subseq s (+ nullable (length "__NULLABLE__")))))))
    (cond
      ((string= s "void") nil)
      ((search "(*" s)
       (let* ((p (+ 2 (search "(*" s)))
              (e (position-if-not #'identifier-char-p s :start p)))
         (make-param :name (subseq s p e) :base "funcptr" :funcptr t)))
      (t
       (let* ((tokens (tokenize-type s))
              (const (member "const" tokens :test #'string=))
              (tokens (remove "const" tokens :test #'string=))
              (ptr (count "*" tokens :test #'string=))
              (ids (remove "*" tokens :test #'string=)))
         (make-param :name (if (> (length ids) 1) (car (last ids)) nil)
                     :base (first ids) :ptr ptr :const (and const t)
                     :nullable (and nullable t)))))))

(defun parse-typedef (stmt)
  (cond
    ((starts-with "typedef struct" stmt)
     (let* ((open (position #\{ stmt))
            (close (position #\} stmt :from-end t))
            (body (subseq stmt (1+ open) close))
            (name (trim (subseq stmt (1+ close))))
            (fields (remove "" (mapcar (lambda (f) (collapse-ws (trim f))) (split-string body #\;))
                            :test #'string=)))
       (cond ((equal fields '("void* ctx")) (pushnew name *handles* :test #'string=))
             ((equal fields '("void* ctx" "void* map_ctx")) (pushnew name *iterators* :test #'string=))
             (t (push (cons name fields) *structs*)))))
    ((starts-with "typedef enum" stmt)
     (let* ((open (position #\{ stmt))
            (close (position #\} stmt :from-end t))
            (name (trim (subseq stmt (1+ close))))
            (value -1)
            (entries (loop for e in (split-top-commas (subseq stmt (1+ open) close))
                           for eq = (position #\= e)
                           collect (if eq
                                       (cons (trim (subseq e 0 eq))
                                             (setf value (parse-integer (trim (subseq e (1+ eq))))))
                                       (cons (trim e) (incf value))))))
       (push (cons name entries) *enums*)))
    ((search "(*" stmt)
     (let* ((p (+ 2 (search "(*" stmt)))
            (e (position-if-not #'identifier-char-p stmt :start p)))
       (push (subseq stmt p e) *funcptr-types*)))))

(defun parse-function (stmt doc header)
  (let* ((paren (position #\( stmt))
         (name-start (1+ (or (position-if-not #'identifier-char-p stmt :end paren :from-end t) -1)))
         (name (subseq stmt name-start paren))
         (ret (trim (subseq stmt 0 name-start)))
         (params-str (subseq stmt (1+ paren) (position #\) stmt :from-end t))))
    (unless (or (search "..." params-str) (starts-with "static" ret))
      (let ((clean-proto (let ((p (search "__NULLABLE__" stmt)))
                           (loop while p
                                 do (setf stmt (concatenate 'string (subseq stmt 0 p) "/* may be null */"
                                                            (subseq stmt (+ p 12)))
                                          p (search "__NULLABLE__" stmt :start2 (+ p 17))))
                           stmt)))
        (push (make-cfun :name name
                         :ret (let ((toks (tokenize-type ret)))
                                (make-param :base (car (last (remove "*" (remove "const" toks :test #'string=)
                                                                     :test #'string=)))
                                            :ptr (count "*" toks :test #'string=)
                                            :const (and (member "const" toks :test #'string=) t)))
                         :params (remove nil (mapcar #'parse-param (split-top-commas params-str)))
                         :doc doc :header header :proto clean-proto)
              *functions*)))))

(defun parse-header (path)
  (let* ((header (pathname-name path))
         (text (strip-preprocessor (strip-comments (read-file path)))))
    (dolist (raw (split-statements text))
      (multiple-value-bind (stmt doc) (take-docs raw)
        (cond ((string= stmt "") nil)
              ((starts-with "typedef" stmt) (parse-typedef stmt))
              ((starts-with "static" stmt) nil)
              ((position #\( stmt) (parse-function stmt doc header)))))))

;;; ------------------------------------------------------------------
;;; Low-level CFFI emission

(defparameter *scalar-types*
  '(("int" . ":int") ("bool" . ":bool") ("float" . ":float") ("double" . ":double")
    ("size_t" . ":size") ("uint64_t" . ":uint64") ("int64_t" . ":int64")
    ("uint32_t" . ":uint32") ("int32_t" . ":int32") ("uint16_t" . ":uint16")
    ("int16_t" . ":int16") ("uint8_t" . ":uint8") ("int8_t" . ":int8")
    ("uintptr_t" . ":uintptr") ("void" . ":void") ("char" . ":char")))

(defun optional-type-p (name) (starts-with "mlx_optional_" name))

(defun cffi-type (p &key return)
  (let ((base (param-base p)) (ptr (param-ptr p)))
    (cond
      ((param-funcptr p) ":pointer")
      ((and (= ptr 1) (string= base "char") (param-const p)) ":string")
      ((plusp ptr) ":pointer")
      ((assoc base *scalar-types* :test #'string=) (cdr (assoc base *scalar-types* :test #'string=)))
      ((member base *handles* :test #'string=) (lispify base))
      ((member base *iterators* :test #'string=) (lispify base))
      ((assoc base *enums* :test #'string=) (lispify base))
      ((optional-type-p base) (lispify base))
      ;; Composites larger than 16 bytes are passed by reference to a
      ;; caller-owned copy under AAPCS64.
      ((string= base "mlx_io_vtable") (if return (error "vtable return") ":pointer"))
      ((member base *funcptr-types* :test #'string=) ":pointer")
      (t (error "Unknown C type ~S" base)))))

(defparameter *reserved-names* '("t" "nil" "pi"))

(defun param-lisp-name (p index)
  (let ((n (if (param-name p) (lispify (param-name p)) (format nil "ARG~D" index))))
    (if (member n *reserved-names* :test #'string=)
        (concatenate 'string n "-arg")
        n)))

(defun enum-keywords (entries)
  "Strip the longest common '_'-terminated prefix from the constant names."
  (let* ((names (mapcar #'car entries))
         (prefix (reduce (lambda (a b) (subseq a 0 (or (mismatch a b) (length a)))) names))
         (prefix (subseq prefix 0 (1+ (or (position #\_ prefix :from-end t) -1)))))
    (mapcar (lambda (e) (cons (lispify (subseq (car e) (length prefix))) (cdr e))) entries)))

(defun doc-string (f)
  (let ((doc (cfun-doc f)) (proto (cfun-proto f)))
    (if doc (format nil "~A~%~%C: ~A" doc proto) (format nil "C: ~A" proto))))

(defun escape (s)
  (with-output-to-string (out)
    (loop for c across s do (when (member c '(#\" #\\)) (write-char #\\ out)) (write-char c out))))

(defun emit-bindings (path)
  (let ((exports '()))
    (with-open-file (out path :direction :output :if-exists :supersede :external-format :utf-8)
      (format out ";;;; GENERATED by tools/generate.lisp from the mlx-c headers -- do not edit.~%")
      (format out ";;;;~%;;;; Raw 1:1 CFFI bindings.  Every mlx-c struct handle is {void* ctx}, which~%")
      (format out ";;;; AAPCS64 (Apple Silicon) passes and returns in a single general register,~%")
      (format out ";;;; exactly like a pointer, so handles are declared as :POINTER aliases.~%")
      (format out ";;;;~%;;;; DEFINE-MLX-FUNCTION (ffi/library.lisp) defines the raw CFFI function as~%")
      (format out ";;;; %NAME and NAME as an inline wrapper that masks float traps.~%~%")
      (format out "(in-package :mlx-ffi)~%~%")
      ;; handles
      (format out ";;; Opaque handle types~%~%")
      (dolist (h (reverse *handles*))
        (push (lispify h) exports)
        (format out "(cffi:defctype ~A :pointer \"struct ~A { void* ctx; }\")~%" (lispify h) h))
      (format out "~%;;; Two-word iterators {ctx, map_ctx}.  Returned in x0/x1; we only receive~%")
      (format out ";;; x0 (ctx).  map_ctx is always the ctx of the map the iterator was made~%")
      (format out ";;; from (verified by disassembly), so callers pass it back explicitly.~%~%")
      (dolist (h (reverse *iterators*))
        (push (lispify h) exports)
        (format out "(cffi:defctype ~A :pointer \"struct ~A { void* ctx; void* map_ctx; } (ctx word only)\")~%"
                (lispify h) h))
      (format out "~%;;; Optionals {T value; bool has_value} are 8-byte non-HFA composites, passed~%")
      (format out ";;; in one GPR: value in bits 0-31, has_value in bits 32-39.~%~%")
      (dolist (s (reverse *structs*))
        (when (optional-type-p (car s))
          (push (lispify (car s)) exports)
          (format out "(cffi:defctype ~A :uint64 \"~A packed as uint64\")~%" (lispify (car s)) (car s))))
      (format out "~%;;; Enums~%~%")
      (dolist (e (reverse *enums*))
        (push (lispify (car e)) exports)
        (format out "(cffi:defcenum ~A~{~%  ~A~})~%~%" (lispify (car e))
                (mapcar (lambda (k) (format nil "(:~A ~D)" (car k) (cdr k))) (enum-keywords (cdr e)))))
      ;; vtable struct
      (let ((vt (assoc "mlx_io_vtable" *structs* :test #'string=)))
        (when vt
          (push "MLX-IO-VTABLE" exports)
          (format out ";;; I/O vtable: a struct of function pointers.~%(cffi:defcstruct mlx-io-vtable~{~%  (~A :pointer)~})~%~%"
                  (mapcar (lambda (f)
                            (let ((p (+ 2 (search "(*" f))))
                              (lispify (subseq f p (position-if-not #'identifier-char-p f :start p)))))
                          (cdr vt)))))
      ;; functions, grouped by header
      (let ((by-header (make-hash-table :test #'equal)) (headers '()))
        (dolist (f (reverse *functions*))
          (unless (gethash (cfun-header f) by-header) (push (cfun-header f) headers))
          (push f (gethash (cfun-header f) by-header)))
        (dolist (h (sort headers #'string<))
          (format out ";;; ------------------------------------------------------------------~%")
          (format out ";;; ~A.h~%~%" h)
          (dolist (f (reverse (gethash h by-header)))
            (let ((lname (lispify (cfun-name f))))
              (push lname exports)
              (format out "(define-mlx-function (\"~A\" ~A) ~A~%  \"~A\""
                      (cfun-name f) lname (cffi-type (cfun-ret f) :return t) (escape (doc-string f)))
              (loop for p in (cfun-params f)
                    for i from 0
                    for pname = (param-lisp-name p i)
                    do (if (member (param-base p) *iterators* :test #'string=)
                           (format out "~%  (~A ~A) (~A-map-ctx :pointer)" pname (cffi-type p) pname)
                           (format out "~%  (~A ~A)" pname (cffi-type p))))
              (format out ")~%~%")))))
      (format out ";;; ------------------------------------------------------------------~%")
      (format out "(eval-when (:compile-toplevel :load-toplevel :execute)~%  (export '(~{~A~^~%            ~})))~%" (mapcar (lambda (s) (format nil "~A" s)) (reverse exports))))
    (length exports)))

;;; ------------------------------------------------------------------
;;; High-level op specs

(defparameter *op-headers* '("ops" "linalg" "fft" "random" "fast" "distributed"))

(defparameter *package-prefixes*
  '(("mlx_linalg_" . "MLX.LINALG") ("mlx_fft_" . "MLX.FFT") ("mlx_random_" . "MLX.RANDOM")
    ("mlx_fast_" . "MLX.FAST") ("mlx_distributed_" . "MLX.DISTRIBUTED") ("mlx_" . "MLX")))

;; C functions whose Lisp name differs from the mechanical one.  A name
;; starting with % is internal (not exported); a handwritten function in
;; src/ provides the public entry point.
(defparameter *renames*
  '(("mlx_arange" . "%ARANGE")
    ("mlx_pad" . "%PAD") ("mlx_pad_symmetric" . "%PAD-SYMMETRIC")
    ("mlx_split" . "%SPLIT") ("mlx_split_sections" . "%SPLIT-SECTIONS")
    ("mlx_tensordot" . "%TENSORDOT") ("mlx_tensordot_axis" . "%TENSORDOT-AXIS")
    ("mlx_linalg_norm" . "NORM-ORD")
    ("mlx_random_split" . "SPLIT-PAIR") ("mlx_random_split_num" . "%SPLIT-NUM")
    ("mlx_random_categorical" . "%CATEGORICAL")
    ("mlx_random_categorical_shape" . "%CATEGORICAL-SHAPE")
    ("mlx_random_categorical_num_samples" . "%CATEGORICAL-NUM-SAMPLES")
    ("mlx_random_permutation" . "%PERMUTATION")
    ("mlx_random_permutation_arange" . "%PERMUTATION-ARANGE")))

;; Axis families: public dispatcher -> variants chosen by the type of the
;; dispatch argument (NIL / integer / list).
;;   (lisp-name dispatch-var default required-p nil-variant int-variant list-variant)
(defparameter *families*
  '(("ALL" "AXIS" "nil" nil "mlx_all" "mlx_all_axis" "mlx_all_axes")
    ("ANY" "AXIS" "nil" nil "mlx_any" "mlx_any_axis" "mlx_any_axes")
    ("ARGMAX" "AXIS" "nil" nil "mlx_argmax" "mlx_argmax_axis" nil)
    ("ARGMIN" "AXIS" "nil" nil "mlx_argmin" "mlx_argmin_axis" nil)
    ("ARGPARTITION" "AXIS" "-1" nil "mlx_argpartition" "mlx_argpartition_axis" nil)
    ("ARGSORT" "AXIS" "-1" nil "mlx_argsort" "mlx_argsort_axis" nil)
    ("CONCATENATE" "AXIS" "0" nil "mlx_concatenate" "mlx_concatenate_axis" nil)
    ("EXPAND-DIMS" "AXIS" nil t nil "mlx_expand_dims" "mlx_expand_dims_axes")
    ("LOGSUMEXP" "AXIS" "nil" nil "mlx_logsumexp" "mlx_logsumexp_axis" "mlx_logsumexp_axes")
    ("MAX" "AXIS" "nil" nil "mlx_max" "mlx_max_axis" "mlx_max_axes")
    ("MEAN" "AXIS" "nil" nil "mlx_mean" "mlx_mean_axis" "mlx_mean_axes")
    ("MIN" "AXIS" "nil" nil "mlx_min" "mlx_min_axis" "mlx_min_axes")
    ("PARTITION" "AXIS" "-1" nil "mlx_partition" "mlx_partition_axis" nil)
    ("PROD" "AXIS" "nil" nil "mlx_prod" "mlx_prod_axis" "mlx_prod_axes")
    ("REPEAT" "AXIS" "nil" nil "mlx_repeat" "mlx_repeat_axis" nil)
    ("ROLL" "AXIS" "nil" nil "mlx_roll" "mlx_roll_axis" "mlx_roll_axes")
    ("SOFTMAX" "AXIS" "nil" nil "mlx_softmax" "mlx_softmax_axis" "mlx_softmax_axes")
    ("SORT" "AXIS" "-1" nil "mlx_sort" "mlx_sort_axis" nil)
    ("SQUEEZE" "AXIS" "nil" nil "mlx_squeeze" "mlx_squeeze_axis" "mlx_squeeze_axes")
    ("STACK" "AXIS" "0" nil "mlx_stack" "mlx_stack_axis" nil)
    ("STD" "AXIS" "nil" nil "mlx_std" "mlx_std_axis" "mlx_std_axes")
    ("SUM" "AXIS" "nil" nil "mlx_sum" "mlx_sum_axis" "mlx_sum_axes")
    ("TAKE" "AXIS" "nil" nil "mlx_take" "mlx_take_axis" nil)
    ("TOPK" "AXIS" "-1" nil "mlx_topk" "mlx_topk_axis" nil)
    ("TRANSPOSE" "AXES" "nil" nil "mlx_transpose" nil "mlx_transpose_axes")
    ("VAR" "AXIS" "nil" nil "mlx_var" "mlx_var_axis" "mlx_var_axes")))

;; Default values (Lisp source text) by parameter name.  A parameter with a
;; default becomes a &KEY argument; one without is a required positional.
;; (:computed "form") means: keyword, and when not supplied, FORM is
;; evaluated after all other arguments are bound.
(defparameter *param-defaults*
  '(("keepdims" . "nil") ("reverse" . "nil") ("inclusive" . "t") ("equal_nan" . "nil")
    ("rtol" . "1d-5") ("atol" . "1d-8") ("precise" . "nil") ("ddof" . "0")
    ("decimals" . "0") ("allow_col_major" . "nil") ("start_axis" . "0") ("end_axis" . "-1")
    ("stride" . "1") ("padding" . "0") ("dilation" . "1") ("groups" . "1")
    ("output_padding" . "0")
    ("stride_0" . "1") ("stride_1" . "1") ("stride_2" . "1")
    ("padding_0" . "0") ("padding_1" . "0") ("padding_2" . "0")
    ("dilation_0" . "1") ("dilation_1" . "1") ("dilation_2" . "1")
    ("output_padding_0" . "0") ("output_padding_1" . "0") ("output_padding_2" . "0")
    ("flip" . "nil") ("alpha" . "1.0") ("beta" . "1.0") ("sorted_indices" . "nil")
    ("sparse" . "nil") ("indexing" . "\"xy\"") ("inverted" . "nil")
    ("upper" . "nil") ("UPLO" . "\"L\"") ("compute_uv" . "t")
    ("loc" . "0.0") ("scale" . "1.0") ("width" . "4") ("eps" . "1e-5")
    ("traditional" . "nil") ("mask_mode" . "\"\"") ("norm" . ":backward")
    ("d" . "1d0") ("num" . "50") ("block_size" . "64") ("nan" . "0.0")
    ("dtype" . ":float32")))

;; Per-function overrides: (c-name param default) where default NIL means
;; "required", a string is a default form, (:computed "form") is computed.
(defparameter *function-defaults*
  '(("mlx_astype" "dtype" nil) ("mlx_view" "dtype" nil) ("mlx_from_fp8" "dtype" nil)
    ("mlx_full_like" "dtype" (:computed "(dtype-of a)"))
    ("mlx_full" "dtype" (:computed "(dtype-of vals)"))
    ("mlx_trace" "dtype" (:computed "(dtype-of a)"))
    ("mlx_trace" "offset" "0") ("mlx_trace" "axis1" "0") ("mlx_trace" "axis2" "1")
    ("mlx_diagonal" "offset" "0") ("mlx_diagonal" "axis1" "0") ("mlx_diagonal" "axis2" "1")
    ("mlx_diag" "k" "0") ("mlx_tril" "k" "0") ("mlx_triu" "k" "0")
    ("mlx_eye" "m" (:computed "n")) ("mlx_eye" "k" "0")
    ("mlx_tri" "m" (:computed "n")) ("mlx_tri" "k" "0")
    ("mlx_as_strided" "offset" "0")
    ("mlx_number_of_elements" "dtype" ":int32")
    ("mlx_slice" "strides" (:computed "(make-list (length start) :initial-element 1)"))
    ("mlx_slice_update" "strides" (:computed "(make-list (length start) :initial-element 1)"))
    ("mlx_slice_update_add" "strides" (:computed "(make-list (length start) :initial-element 1)"))
    ("mlx_slice_update_max" "strides" (:computed "(make-list (length start) :initial-element 1)"))
    ("mlx_slice_update_min" "strides" (:computed "(make-list (length start) :initial-element 1)"))
    ("mlx_slice_update_prod" "strides" (:computed "(make-list (length start) :initial-element 1)"))
    ("mlx_quantized_matmul" "transpose" "t") ("mlx_gather_qmm" "transpose" "t")
    ("mlx_quantize" "mode" "\"affine\"") ("mlx_dequantize" "mode" "\"affine\"")
    ("mlx_quantized_matmul" "mode" "\"affine\"") ("mlx_gather_qmm" "mode" "\"affine\"")
    ("mlx_qqmm" "mode" "\"nvfp4\"")
    ("mlx_pad" "mode" "\"constant\"") ("mlx_pad" "pad_value" "0")
    ("mlx_pad_symmetric" "mode" "\"constant\"") ("mlx_pad_symmetric" "pad_value" "0")
    ;; fft
    ("mlx_fft_fft" "axis" "-1") ("mlx_fft_ifft" "axis" "-1") ("mlx_fft_rfft" "axis" "-1")
    ("mlx_fft_irfft" "axis" "-1")
    ("mlx_fft_fft" "n" (:computed "(dim a axis)")) ("mlx_fft_ifft" "n" (:computed "(dim a axis)"))
    ("mlx_fft_rfft" "n" (:computed "(dim a axis)"))
    ("mlx_fft_irfft" "n" (:computed "(* 2 (1- (dim a axis)))"))
    ("mlx_fft_fft2" "axes" "'(-2 -1)") ("mlx_fft_ifft2" "axes" "'(-2 -1)")
    ("mlx_fft_rfft2" "axes" "'(-2 -1)") ("mlx_fft_irfft2" "axes" "'(-2 -1)")
    ("mlx_fft_fftn" "axes" (:computed "(all-axes a)")) ("mlx_fft_ifftn" "axes" (:computed "(all-axes a)"))
    ("mlx_fft_rfftn" "axes" (:computed "(all-axes a)")) ("mlx_fft_irfftn" "axes" (:computed "(all-axes a)"))
    ("mlx_fft_fft2" "n" (:computed "(axes-dims a axes)")) ("mlx_fft_ifft2" "n" (:computed "(axes-dims a axes)"))
    ("mlx_fft_rfft2" "n" (:computed "(axes-dims a axes)"))
    ("mlx_fft_irfft2" "n" (:computed "(axes-dims a axes :inverse-real t)"))
    ("mlx_fft_fftn" "n" (:computed "(axes-dims a axes)")) ("mlx_fft_ifftn" "n" (:computed "(axes-dims a axes)"))
    ("mlx_fft_rfftn" "n" (:computed "(axes-dims a axes)"))
    ("mlx_fft_irfftn" "n" (:computed "(axes-dims a axes :inverse-real t)"))
    ("mlx_fft_fftshift" "axes" (:computed "(all-axes a)"))
    ("mlx_fft_ifftshift" "axes" (:computed "(all-axes a)"))
    ;; linalg
    ("mlx_linalg_cross" "axis" "-1")
    ;; random
    ("mlx_random_bernoulli" "p" "0.5") ("mlx_random_bernoulli" "shape" (:computed "(shape-of p)"))
    ("mlx_random_bits" "shape" "'()") ("mlx_random_gumbel" "shape" "'()")
    ("mlx_random_laplace" "shape" "'()") ("mlx_random_normal" "shape" "'()")
    ("mlx_random_normal_broadcast" "shape" "'()")
    ("mlx_random_uniform" "shape" "'()") ("mlx_random_uniform" "low" "0.0") ("mlx_random_uniform" "high" "1.0")
    ("mlx_random_randint" "shape" "'()") ("mlx_random_randint" "dtype" ":int32")
    ("mlx_random_truncated_normal" "shape" "'()")
    ("mlx_random_multivariate_normal" "shape" "'()")
    ("mlx_random_categorical" "axis" "-1") ("mlx_random_categorical_shape" "axis" "-1")
    ("mlx_random_categorical_num_samples" "axis" "-1")
    ("mlx_random_permutation" "axis" "0")
    ("mlx_random_split_num" "num" "2")
    ;; fast
    ("mlx_fast_rope" "offset" "0")
    ("mlx_fast_scaled_dot_product_attention" "scale" nil)
    ;; generic names that mean something else in these functions
    ("mlx_random_truncated_normal" "upper" nil)
    ("mlx_hadamard_transform" "scale" "nil")
    ("mlx_dequantize" "dtype" "nil")
    ("mlx_random_normal_broadcast" "loc" "nil") ("mlx_random_normal_broadcast" "scale" "nil")
    ("mlx_conv_general" "stride" nil)
    ("mlx_tri" "type" ":float32")
    ;; ops whose parameter names collide with defaults above
    ("mlx_fft_fftfreq" "n" nil) ("mlx_fft_rfftfreq" "n" nil)
    ("mlx_eye" "n" nil) ("mlx_tri" "n" nil) ("mlx_identity" "n" nil)))

;; C parameter names replaced in the Lisp lambda list: ((c-fn . c-param) . lisp-name)
(defparameter *param-renames*
  '((("mlx_tri" . "type") . "dtype")))

(defun lisp-param-name (fname cname)
  (param-lisp-name (make-param :name (or (cdr (assoc (cons fname cname) *param-renames* :test #'equal))
                                         cname))
                   0))

(defun param-default (fname pname)
  (let ((ov (find-if (lambda (e) (and (string= (first e) fname) (string= (second e) pname)))
                     *function-defaults*)))
    (if ov (third ov) (cdr (assoc pname *param-defaults* :test #'string=)))))

(defun high-level-name (cname)
  "Returns (values package-name symbol-name)."
  (let* ((entry (find-if (lambda (e) (starts-with (car e) cname)) *package-prefixes*))
         (pkg (cdr entry))
         (rename (cdr (assoc cname *renames* :test #'string=))))
    (values pkg (or rename (lispify (subseq cname (length (car entry))))))))

(defun qualified (pkg name)
  (if (starts-with "%" name)
      (format nil "~(~A~)" name)          ; internal: lives in MLX.IMPL
      (format nil "~(~A:~A~)" pkg name)))

(defun classify-op (f)
  "Return (values outs args stream-p) or NIL if F doesn't fit the op pattern."
  (let ((params (cfun-params f)) (outs '()) (args '()) (stream-p nil))
    (unless (and (string= (param-base (cfun-ret f)) "int") (zerop (param-ptr (cfun-ret f))))
      (return-from classify-op nil))
    ;; leading outputs
    (loop while (and params (= (param-ptr (first params)) 1) (not (param-const (first params)))
                     (member (param-base (first params)) '("mlx_array" "mlx_vector_array") :test #'string=)
                     (starts-with "res" (param-name (first params))))
          do (push (if (string= (param-base (pop params)) "mlx_array") :array :arrays) outs))
    (when (null outs) (return-from classify-op nil))
    ;; trailing stream
    (let ((last (car (last params))))
      (when (and last (string= (param-base last) "mlx_stream"))
        (setf stream-p t params (butlast params))))
    (loop while params
          do (let* ((p (pop params)) (base (param-base p)) (ptr (param-ptr p)) (name (param-name p))
                    (kind
                      (cond
                        ((and (string= base "mlx_array") (zerop ptr)) (if (param-nullable p) :array-or-null :array))
                        ((and (string= base "mlx_vector_array") (zerop ptr)) :arrays)
                        ((and (member base '("int" "int64_t") :test #'string=) (= ptr 1) (param-const p)
                              params (string= (param-base (first params)) "size_t"))
                         (pop params)
                         (cond ((string= base "int64_t") :int64s)
                               ((param-nullable p) :ints-or-null)
                               (t :ints)))
                        ((plusp ptr)
                         (if (and (string= base "char") (= ptr 1)) :string (return-from classify-op nil)))
                        ((string= base "int") :int)
                        ((string= base "size_t") :size)
                        ((string= base "uint64_t") :uint64)
                        ((string= base "bool") :bool)
                        ((string= base "float") :float)
                        ((string= base "double") :double)
                        ((string= base "mlx_dtype") :dtype)
                        ((string= base "mlx_optional_int") :optional-int)
                        ((string= base "mlx_optional_float") :optional-float)
                        ((string= base "mlx_optional_dtype") :optional-dtype)
                        ((string= base "mlx_fft_norm") :fft-norm)
                        ((string= base "mlx_distributed_group") :group-or-null)
                        (t (return-from classify-op nil)))))
               (push (list name kind) args)))
    (values (nreverse outs) (nreverse args) stream-p)))

(defun arg-spec (fname arg)
  "Spec text for one argument: (VAR KIND [:default form | :computed form])."
  (destructuring-bind (cname kind) arg
    (let* ((var (lisp-param-name fname cname))
           (default (cond ((member kind '(:array-or-null :ints-or-null :optional-int :optional-float
                                          :optional-dtype :group-or-null))
                           (or (param-default fname cname) "nil"))
                          (t (param-default fname cname)))))
      (cond ((null default) (format nil "(~(~A ~S~))" var kind))
            ((consp default) (format nil "(~(~A ~S~) :computed ~A)" var kind (second default)))
            (t (format nil "(~(~A ~S~) :default ~A)" var kind default))))))

(defvar *op-info* (make-hash-table :test #'equal)) ; cname -> (lisp-sym args)

(defun emit-ops (path)
  (let ((exports (make-hash-table :test #'equal)) (count 0) (skipped '()))
    (with-open-file (out path :direction :output :if-exists :supersede :external-format :utf-8)
      (format out ";;;; GENERATED by tools/generate.lisp from the mlx-c headers -- do not edit.~%")
      (format out ";;;;~%;;;; High-level array operations.  Each DEFINE-OP expands into a function that~%")
      (format out ";;;; converts Lisp arguments, calls the raw binding with a fresh result slot,~%")
      (format out ";;;; checks the status code and wraps the result.  See src/op.lisp.~%~%")
      (format out "(in-package :mlx.impl)~%")
      (let ((family-members (loop for fam in *families* append (remove nil (subseq fam 4)))))
        (dolist (f (reverse *functions*))
          (when (member (cfun-header f) *op-headers* :test #'string=)
            (multiple-value-bind (outs args stream-p) (classify-op f)
              (if (null outs)
                  (push (cfun-name f) skipped)
                  (multiple-value-bind (pkg name) (high-level-name (cfun-name f))
                    ;; the NIL-variant of a family is internal
                    (when (and (member (cfun-name f) family-members :test #'string=)
                               (find name *families* :key #'first :test #'string-equal))
                      (setf name (concatenate 'string "%" name)))
                    (unless (starts-with "%" name) (push name (gethash pkg exports)))
                    (setf (gethash (cfun-name f) *op-info*) (list (qualified pkg name) args stream-p))
                    (incf count)
                    (format out "~%(define-op ~A ~A~%  :returns ~(~S~)~%  :args (~{~A~^~%         ~})~%  :stream ~A~%  :doc \"~A\")~%"
                            (qualified pkg name)
                            (format nil "mlx-ffi:~A" (lispify (cfun-name f)))
                            outs
                            (mapcar (lambda (a) (arg-spec (cfun-name f) a)) args)
                            (if stream-p "t" "nil")
                            (escape (doc-string f)))))))))
      ;; families
      (format out "~%~%;;; ------------------------------------------------------------------~%")
      (format out ";;; Axis-family dispatchers: choose a variant by the type of the axis argument.~%")
      (dolist (fam *families*)
        (destructuring-bind (lname dvar default required nil-v int-v list-v) fam
          (let* ((variants (remove nil (list (and nil-v (cons :nil nil-v)) (and int-v (cons :int int-v))
                                             (and list-v (cons :list list-v)))))
                 (infos (mapcar (lambda (v) (cons (car v) (gethash (cdr v) *op-info*))) variants))
                 ;; the base argument list: required positional args shared by all variants
                 (base (or (cdr (assoc :nil infos)) (cdr (first infos))))
                 (base-args (second base))
                 (base-fname (or nil-v int-v))
                 (dispatch-names '("axis" "axes"))
                 (lisp-var (lambda (cname) (string-downcase (param-lisp-name (make-param :name cname) 0))))
                 (positional (loop for (cname nil) in base-args
                                   unless (or (member cname dispatch-names :test #'string=)
                                              (param-default base-fname cname))
                                     collect (funcall lisp-var cname)))
                 (keys '()))
            (dolist (inf infos)
              (loop with fname = (cdr (assoc (car inf) variants))
                    for (cname nil) in (second (cdr inf))
                    for d = (param-default fname cname)
                    for v = (funcall lisp-var cname)
                    when (and d (not (member cname dispatch-names :test #'string=))
                              (not (assoc v keys :test #'string=)))
                      do (push (cons v d) keys)))
            (setf keys (nreverse keys))
            (push lname (gethash "MLX" exports))
            (let ((lambda-list
                    (append positional
                            (when required (list (string-downcase dvar)))
                            (list "&key")
                            (unless required (list (format nil "(~(~A~) ~A)" dvar default)))
                            (mapcar (lambda (k) (if (consp (cdr k))
                                                    (car k)
                                                    (format nil "(~A ~A)" (car k) (cdr k))))
                                    keys)
                            (list "stream"))))
              (format out "~%(defun mlx:~(~A~) (~{~A~^ ~})~%" lname lambda-list)
              (format out "  \"Dispatches on ~(~A~): ~{~A~^, ~}.\"~%"
                      dvar (mapcar (lambda (v) (format nil "~(~A~) -> ~A" (car v) (cdr v))) variants))
              (format out "  (etypecase ~(~A~)" dvar)
              (dolist (inf infos)
                (let* ((sym (first (cdr inf)))
                       (vargs (second (cdr inf)))
                       (fname (cdr (assoc (car inf) variants)))
                       (call-pos (loop for (cname nil) in vargs
                                       unless (param-default fname cname)
                                         collect (if (member cname dispatch-names :test #'string=)
                                                     (string-downcase dvar)
                                                     (funcall lisp-var cname))))
                       (call-keys (loop for (cname nil) in vargs
                                        for v = (funcall lisp-var cname)
                                        when (and (param-default fname cname)
                                                  (not (member cname dispatch-names :test #'string=)))
                                          collect (format nil ":~A ~A" v v))))
                  (format out "~%    (~A (~(~A~)~{ ~A~}~{ ~A~} :stream stream))"
                          (ecase (car inf) (:nil "null") (:int "integer") (:list "list"))
                          sym call-pos call-keys)))
              (format out "))~%")))))
      (terpri out))
    (values count exports skipped)))

(defun emit-exports (path exports)
  (with-open-file (out path :direction :output :if-exists :supersede :external-format :utf-8)
    (format out ";;;; GENERATED by tools/generate.lisp -- do not edit.~%")
    (format out ";;;; Exported symbol names of generated operations, per package.~%(~%")
    (dolist (pkg '("MLX" "MLX.LINALG" "MLX.FFT" "MLX.RANDOM" "MLX.FAST" "MLX.DISTRIBUTED"))
      (format out " (~S~{~% ~S~})~%" pkg (sort (remove-duplicates (mapcar #'string-upcase (gethash pkg exports))
                                                                  :test #'string=)
                                               #'string<)))
    (format out ")~%")))

;;; ------------------------------------------------------------------

(defun find-include-dir (args)
  "The include/mlx/c directory: the one given (which must exist), else
$MLX_C_INCLUDE, /opt/homebrew/include/mlx/c or /usr/local/include/mlx/c."
  (flet ((valid (c) (and c (probe-file (concatenate 'string (string-right-trim "/" c) "/ops.h"))
                         (concatenate 'string (string-right-trim "/" c) "/"))))
    (cond ((first args)
           (or (valid (first args))
               (error "No mlx-c headers (ops.h) in ~A" (first args))))
          (t (or (some #'valid (list (sb-ext:posix-getenv "MLX_C_INCLUDE")
                                     "/opt/homebrew/include/mlx/c/"
                                     "/usr/local/include/mlx/c/"))
                 (error "Cannot find mlx-c headers; pass the include/mlx/c directory."))))))

;;; ------------------------------------------------------------------
;;; ABI validation
;;;
;;; The bindings pass structs by value using knowledge of their layout
;;; (see the header of the generated bindings).  If mlx-c ever changes a
;;; struct, generating silently would produce bindings that corrupt memory,
;;; so every struct and the dtype enum must match what the code expects.

(defparameter *expected-optionals*
  '(("mlx_optional_int" "int value" "bool has_value")
    ("mlx_optional_float" "float value" "bool has_value")
    ("mlx_optional_dtype" "mlx_dtype value" "bool has_value"))
  "Packed into a uint64 by PACK-OPTIONAL-* in src/dtype.lisp.")

(defparameter *expected-vtable*
  '("mlx_io_vtable"
    "bool (*is_open)(void*)" "bool (*good)(void*)" "size_t (*tell)(void*)"
    "void (*seek)(void*, int64_t off, int whence)" "void (*read)(void*, char* data, size_t n)"
    "void (*read_at_offset)(void*, char* data, size_t n, size_t off)"
    "void (*write)(void*, const char* data, size_t n)" "const char* (*label)(void*)"
    "void (*free)(void*)")
  "Filled field by field in src/io.lisp.")

(defparameter *expected-dtypes*
  '("MLX_BOOL" "MLX_UINT8" "MLX_UINT16" "MLX_UINT32" "MLX_UINT64" "MLX_INT8" "MLX_INT16"
    "MLX_INT32" "MLX_INT64" "MLX_FLOAT16" "MLX_FLOAT32" "MLX_FLOAT64" "MLX_BFLOAT16"
    "MLX_COMPLEX64")
  "Mirrored by +DTYPES+ in src/dtype.lisp.")

(defun validate-abi ()
  (let ((problems '()))
    (flet ((problem (fmt &rest args) (push (apply #'format nil fmt args) problems)))
      (dolist (s *structs*)
        (let ((expected (or (assoc (car s) *expected-optionals* :test #'string=)
                            (and (string= (car s) "mlx_io_vtable") *expected-vtable*))))
          (cond ((null expected)
                 (problem "struct ~A {~{~A;~^ ~}} has an unknown layout" (car s) (cdr s)))
                ((not (equal (cdr s) (cdr expected)))
                 (problem "struct ~A changed:~%    expected {~{~A;~^ ~}}~%    found    {~{~A;~^ ~}}"
                          (car s) (cdr expected) (cdr s))))))
      (dolist (e *expected-optionals*)
        (unless (assoc (car e) *structs* :test #'string=)
          (problem "struct ~A is missing" (car e))))
      (let ((dtypes (cdr (assoc "mlx_dtype" *enums* :test #'string=))))
        (unless (and (equal (mapcar #'car dtypes) *expected-dtypes*)
                     (equal (mapcar #'cdr dtypes) (loop for i below (length dtypes) collect i)))
          (problem "enum mlx_dtype changed: ~S" dtypes))))
    (when problems
      (format *error-output* "~&ABI check failed -- the bindings' by-value struct handling needs~%~
                              review before regenerating (see tools/generate.lisp):~%~{  - ~A~%~}"
              (reverse problems))
      (sb-ext:exit :code 2))))

;;; ------------------------------------------------------------------

(defparameter *outputs* '("src/ffi/bindings.lisp" "src/generated/ops.lisp" "src/generated/exports.sexp"))

(defun generate-into (dir)
  "Write all generated files under DIR; returns (values bindings ops skipped)."
  (ensure-directories-exist (merge-pathnames "src/ffi/" dir))
  (ensure-directories-exist (merge-pathnames "src/generated/" dir))
  (let ((bindings (emit-bindings (merge-pathnames "src/ffi/bindings.lisp" dir))))
    (multiple-value-bind (count exports skipped) (emit-ops (merge-pathnames "src/generated/ops.lisp" dir))
      (emit-exports (merge-pathnames "src/generated/exports.sexp" dir) exports)
      (values bindings count skipped))))

(defun main (args)
  (let* ((check (member "--check" args :test #'string=))
         (dir (find-include-dir (remove "--check" args :test #'string=)))
         (root (make-pathname :name nil :type nil
                              :defaults (merge-pathnames "../" (make-pathname :name nil :type nil
                                                                              :defaults *load-truename*)))))
    (format t "Reading headers from ~A~%" dir)
    (dolist (h (sort (mapcar #'namestring (directory (concatenate 'string dir "*.h"))) #'string<))
      (parse-header h))
    (format t "Parsed ~D functions, ~D handle types, ~D enums~%"
            (length *functions*) (length *handles*) (length *enums*))
    (validate-abi)
    (if check
        (let* ((tmp (pathname (format nil "~A/mlx-generate-check-~D/"
                                      (string-right-trim "/" (or (sb-ext:posix-getenv "TMPDIR") "/tmp"))
                                      (sb-posix:getpid))))
               (stale (progn (generate-into tmp)
                             (remove-if (lambda (f)
                                          (let ((committed (merge-pathnames f root)))
                                            (and (probe-file committed)
                                                 (string= (read-file committed)
                                                          (read-file (merge-pathnames f tmp))))))
                                        *outputs*))))
          (if stale
              (progn (format t "Out of date with the installed headers: ~{~A~^, ~}~%~
                                Run `make generate`, review the diff, and run the tests.~%" stale)
                     (sb-ext:exit :code 1))
              (format t "Generated files are up to date.~%")))
        (multiple-value-bind (bindings count skipped) (generate-into root)
          (format t "Wrote ~D raw bindings~%Wrote ~D op specs~%" bindings count)
          (format t "Not auto-wrapped (handwritten in src/): ~{~A~^, ~}~%" (reverse skipped))))))

(main (rest sb-ext:*posix-argv*))
