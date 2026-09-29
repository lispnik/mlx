;;;; complete.lisp -- code completion for the editor, informed by the live image
;;;;
;;;; COMPLETE-LISP fills in code at point with a fill-in-the-middle model.
;;;; Two things make it a Lisp completer rather than a text completer:
;;;;
;;;;  - The prompt carries what the running image knows: the lambda lists
;;;;    and docstrings of the functions, macros and variables used near point,
;;;;    and the other definitions in the current package, so the model calls
;;;;    real functions with real arguments.
;;;;  - Decoding follows the reader (see lisp.lisp) from the state of the
;;;;    top-level form being edited, so a completion never breaks the form's
;;;;    structure, and stops when the form closes.
;;;;
;;;; Afterwards the completion is checked against the image: operators it
;;;; calls that are not defined anywhere are reported.
;;;;
;;;; EMACS-COMPLETE is the entry point for emacs/mlx-complete.el, called
;;;; through SLY or SLIME.

(in-package :mlx.llm)

(defparameter *completion-model-name* "mlx-community/Qwen2.5-Coder-1.5B-4bit"
  "The model EMACS-COMPLETE loads: a base (not instruct) coder model trained
for fill-in-the-middle.")

(defvar *completion-model* nil)
(defvar *completion-lock* (sb-thread:make-mutex :name "mlx completion"))

(defun completion-model ()
  (or *completion-model* (setf *completion-model* (load-model *completion-model-name*))))

;;; ------------------------------------------------------------------
;;; What the image knows

(defun symbol-tokens (text)
  "The symbol-like tokens of TEXT, in order, without duplicates.  Strings,
comments and character literals are skipped."
  (let ((tokens '()) (i 0) (n (length text)))
    (flet ((delimiter-p (c) (or (member c '(#\( #\) #\' #\` #\, #\" #\; #\Space #\Tab #\Newline #\Return #\Page)))))
      (loop while (< i n)
            do (let ((c (char text i)))
                 (cond ((char= c #\") (setf i (string-end text (1+ i))))
                       ((char= c #\;) (setf i (or (position #\Newline text :start i) n)))
                       ((and (char= c #\#) (< (1+ i) n) (char= (char text (1+ i)) #\\)) (incf i 3))
                       ((and (char= c #\#) (< (1+ i) n) (char= (char text (1+ i)) #\|))
                        (setf i (let ((e (search "|#" text :start2 i))) (if e (+ e 2) n))))
                       ((delimiter-p c) (incf i))
                       (t (let ((end (or (position-if #'delimiter-p text :start i) n)))
                            (push (subseq text i end) tokens)
                            (setf i end)))))))
    (remove-duplicates (nreverse tokens) :test #'string-equal :from-end t)))

(defun string-end (text start)
  "The index after the string whose contents begin at START."
  (do ((j start (1+ j)))
      ((>= j (length text)) j)
    (case (char text j)
      (#\\ (incf j))
      (#\" (return (1+ j))))))

(defun resolve-token (token package)
  "The existing symbol TOKEN names when read in PACKAGE, or NIL."
  (let* ((colon (position #\: token))
         (name (string-upcase (if colon (string-left-trim ":" (subseq token colon)) token)))
         (home (cond ((null colon) package)
                     ((zerop colon) (find-package :keyword))
                     (t (find-package (string-upcase (subseq token 0 colon)))))))
    (and home (plusp (length name)) (not (every (lambda (c) (or (digit-char-p c) (find c "+-./"))) name))
         (values (find-symbol name home)))))

(defun first-line (string &optional (limit 100))
  (let ((line (subseq string 0 (or (position #\Newline string) (length string)))))
    (if (> (length line) limit) (concatenate 'string (subseq line 0 (- limit 3)) "...") line)))

(defun describe-for-context (symbol package)
  "A one-line comment describing SYMBOL's definition, or NIL."
  (let ((*package* package) (*print-case* :downcase) (*print-pretty* nil))
    (flet ((doc (type) (let ((d (documentation symbol type))) (and d (first-line d)))))
      (cond ((special-operator-p symbol) nil)
            ((macro-function symbol)
             (format nil ";; macro ~S~@[ -- ~A~]" (cons symbol (sb-introspect:function-lambda-list symbol))
                     (doc 'function)))
            ((fboundp symbol)
             (format nil ";; ~:[function~;generic function~] ~S~@[ -- ~A~]"
                     (typep (fdefinition symbol) 'generic-function)
                     (cons symbol (sb-introspect:function-lambda-list symbol)) (doc 'function)))
            ((boundp symbol)
             (format nil ";; variable ~S~@[ -- ~A~]" symbol (doc 'variable)))
            ((find-class symbol nil)
             (format nil ";; class ~S~@[ -- ~A~]" symbol (doc 'type)))))))

(defun image-context (text package &key (limit 60) (others 100))
  "Comments describing, from the live image, the definitions TEXT uses
(outside COMMON-LISP), then PACKAGE's own definitions: fully up to LIMIT
lines in all, then by name up to OTHERS more."
  (let* ((cl (find-package :common-lisp))
         (used (loop for token in (symbol-tokens text)
                     for s = (resolve-token token package)
                     when (and s (not (eq (symbol-package s) cl)) (not (keywordp s)))
                       collect s))
         (own (let ((symbols '()))
                (do-symbols (s package)
                  (when (and (eq (symbol-package s) package) (or (fboundp s) (boundp s) (find-class s nil)))
                    (pushnew s symbols)))
                (sort symbols #'string<)))
         (lines '()) (described '()) (named '()))
    (dolist (s (append used (set-difference own used)))
      (let ((line (and (< (length lines) limit) (describe-for-context s package))))
        (cond (line (push line lines) (push s described))
              ((and (member s own) (< (length named) others)) (push s named)))))
    (let ((*package* package) (*print-case* :downcase))
      (format nil ";;; The running Lisp image.  Current package: ~A~@[ (uses ~{~A~^, ~})~].~%~
                   ~@[;;; Its definitions, those used near point first:~%~{~A~%~}~]~
                   ~@[;;; Also defined: ~{~S~^ ~}~%~]"
              (package-name package) (mapcar #'package-name (package-use-list package))
              (reverse lines) (reverse named)))))

;;; ------------------------------------------------------------------
;;; The prompt and decoding

(defun completion-prompt (context form-prefix suffix package file)
  "A Qwen2.5-Coder repository-level fill-in-the-middle prompt: the image
context as one file, then the buffer with the hole at point."
  (format nil "<|repo_name|>lisp-image~%<|file_sep|>image-context.lisp~%~A~%<|file_sep|>~A~%~
               <|fim_prefix|>~A~A<|fim_suffix|>~A<|fim_middle|>"
          (image-context (concatenate 'string context form-prefix) package)
          (or file "buffer.lisp") context form-prefix suffix))

(defun text-before-close (lexer octets)
  "If OCTETS (a token) would complete LEXER's form, the text of the part
before the byte that completes it (as a string), else NIL."
  (let ((copy (copy-lisp-lexer lexer)))
    (loop for b across octets for i from 0
          do (unless (lex-byte copy b) (return nil))
             (when (lisp-form-complete-p copy)
               (return (sb-ext:octets-to-string (subseq octets 0 i) :external-format :utf-8))))))

(defun form-lexer (form-prefix)
  "A lexer positioned after FORM-PREFIX (text from the ( that opens the
top-level form to point), or NIL when it isn't one."
  (let ((lexer (make-lisp-lexer)))
    (and (plusp (length form-prefix))
         (lex-octets lexer (sb-ext:string-to-octets form-prefix :external-format :utf-8))
         (not (lisp-form-complete-p lexer))
         lexer)))

(defun complete-lisp (model form-prefix &key (context "") (suffix "") (package *package*) file
                                             (mode :form) (max-tokens 128))
  "Complete the Lisp at point.  FORM-PREFIX is the text from the start of
the top-level form being edited to point (\"\" at top level), CONTEXT the
text before that and SUFFIX the text after point.  MODE :FORM runs until
the model stops or the form closes; :LINE also stops at a newline.
Decoding follows the reader, so the form stays well formed.  Returns
(values completion complaints): what the compiler, in this image, says is
undefined in the completed form."
  (let* ((tokenizer (model-tokenizer model))
         (package (or (find-package package) (find-package :cl-user)))
         (constraint (tokenizer-constraint tokenizer))
         (special (added-token-ids tokenizer))
         (lexer (or (form-lexer form-prefix) (and (zerop (length (string-trim '(#\Space #\Newline) form-prefix)))
                                                  (make-lisp-lexer))))
         ;; closers right after point (an editor's paired parens) belong to
         ;; the form: the completion must leave them to close it
         (reserved (count #\) (subseq suffix 0 (or (position-if-not (lambda (c) (find c ") 	")) suffix)
                                                   (length suffix)))))
         (closed-by-suffix (and lexer (>= reserved (lisp-lexer-depth lexer)) (plusp reserved)))
         ;; token healing: a prompt ending in spaces tokenizes unnaturally, so
         ;; the model writes them again and they are removed afterwards
         (healed (let ((last (position-if-not (lambda (c) (member c '(#\Space #\Tab))) form-prefix
                                              :from-end t)))
                   (subseq form-prefix (if last (1+ last) 0))))
         (prompt (encode tokenizer (completion-prompt context (subseq form-prefix 0 (- (length form-prefix) (length healed)))
                                                      suffix package file)))
         (cache (make-cache model))
         (ids '()) (vocab nil) (tail nil))
    (when (and lexer (not closed-by-suffix) (plusp reserved))
      (decf (lisp-lexer-depth lexer) reserved))
    (mx:with-scope ()
      (funcall model (mx:from-lisp (list (butlast prompt)) :dtype :int32) cache))
    (let ((input (last prompt)))
      (loop repeat max-tokens
            do (let ((id (mx:with-scope ()
                           (let* ((logits (mx:ref (funcall model (mx:from-lisp (list input) :dtype :int32) cache) t -1))
                                  (unconstrained (mx:item (mx:argmax logits :axis -1))))
                             (setf vocab (or vocab (mx:dim logits -1)))
                             (cond ((or (gethash unconstrained special)
                                        (member unconstrained (eos-tokens tokenizer)))
                                    nil)
                                   ;; the model means to close the form, but the
                                   ;; text after point does that: it is done
                                   ((and closed-by-suffix
                                         (setf tail (text-before-close lexer (svref (lisp-constraint-octets constraint)
                                                                                    unconstrained))))
                                    nil)
                                   (lexer (mx:item (mx:argmax (mx:add logits (lisp-mask constraint lexer vocab
                                                                                        :forbid-close closed-by-suffix))
                                                              :axis -1)))
                                   (t unconstrained))))))
                 (unless id (return))
                 (when lexer (lex-octets lexer (svref (lisp-constraint-octets constraint) id)))
                 (push id ids)
                 (setf input (list id))
                 (when (or (and lexer (lisp-form-complete-p lexer))
                           (and (eq mode :line) (find #\Newline (decode tokenizer (reverse ids)))))
                   (return)))))
    (let* ((text (concatenate 'string (decode tokenizer (reverse ids)) (or tail "")))
           (text (subseq text (let ((n (min (length healed) (length text))))
                                (or (mismatch healed text :end1 n :end2 n) n))))
           (text (if (eq mode :line) (subseq text 0 (or (position #\Newline text) (length text))) text))
           (text (if (and lexer (lisp-form-complete-p lexer)) (string-right-trim '(#\Space #\Tab #\Newline) text) text)))
      (values text (let ((form (completed-form form-prefix text suffix)))
                     (and form (compiler-complaints form package)))))))

(defun compiler-complaints (text package)
  "Compile the form TEXT (read in PACKAGE, without evaluating anything) and
return what the compiler says is undefined, e.g. (\"undefined function:
frob\").  Symbols the reader had to create are uninterned afterwards."
  (let* ((*package* package)
         (fresh (remove-if (lambda (token) (resolve-token token package)) (symbol-tokens text)))
         (complaints '()))
    (unwind-protect
         (let ((form (handler-case (let ((*read-eval* nil)) (read-from-string text))
                       (error () (return-from compiler-complaints nil)))))
           (handler-bind ((warning (lambda (w)
                                     (let ((message (let ((*print-case* :downcase)) (princ-to-string w))))
                                       (when (search "undefined" message)
                                         (push (substitute #\Space #\Newline message) complaints)))
                                     (muffle-warning w))))
             (with-compilation-unit (:override t)
               (ignore-errors (compile nil `(lambda () ,form))))))
      (dolist (token fresh)
        (let ((s (find-symbol (string-upcase token) package)))
          (when (and s (eq (symbol-package s) package)) (unintern s package)))))
    (remove-duplicates (nreverse complaints) :test #'string=)))

(defun completed-form (form-prefix completion suffix)
  "The text of the top-level form once COMPLETION is inserted, if it closes."
  (let ((lexer (make-lisp-lexer))
        (text (concatenate 'string form-prefix completion suffix)))
    (loop for c across text for i from 0
          do (unless (lex-octets lexer (sb-ext:string-to-octets (string c) :external-format :utf-8))
               (return nil))
             (when (lisp-form-complete-p lexer)
               (return (subseq text 0 (1+ i)))))))

;;; ------------------------------------------------------------------
;;; Editor entry point

(defun emacs-complete (form-prefix context suffix package-name file mode)
  "For emacs/mlx-complete.el: complete at point with *COMPLETION-MODEL*.
Returns (completion complaints seconds), or (\"\" nil seconds
error-message)."
  (let ((start (get-internal-real-time)))
    (flet ((elapsed () (float (/ (- (get-internal-real-time) start) internal-time-units-per-second))))
      (handler-case
          (sb-thread:with-mutex (*completion-lock*)
            (multiple-value-bind (text unknown)
                (complete-lisp (completion-model) form-prefix :context context :suffix suffix
                                                              :package (or (find-package (string-upcase package-name))
                                                                           (find-package package-name)
                                                                           :cl-user)
                                                              :file file :mode (if (eq mode :line) :line :form))
              (list text unknown (elapsed))))
        (error (e) (list "" nil (elapsed) (princ-to-string e)))))))
