;;;; llm/lisp.lisp -- a language model that writes Common Lisp: decoding
;;;; constrained by the Lisp reader, and a write/run/repair loop
;;;;
;;;; 1. Constrained decoding.  A small lexer follows the Lisp reader's
;;;;    view of the text generated so far (paren depth, strings, |symbols|,
;;;;    comments, escapes, #\ characters).  Before each token is sampled,
;;;;    every token that would break the form -- a ")" with nothing open,
;;;;    text before the "(" or after the closing ")", an unknown "#"
;;;;    dispatch, read-time evaluation "#." -- is masked out.  The result is
;;;;    exactly one balanced form (unless the token budget runs out).  The
;;;;    reader can still refuse details the lexer does not model (a comma
;;;;    outside a backquote, an unknown package prefix); that is reported to
;;;;    the model like any other error.
;;;;
;;;; 2. Evaluation.  The form is evaluated -- in a separate SBCL process by
;;;;    default, so model-written code cannot touch this image -- along with
;;;;    any test forms, capturing output, warnings and errors.
;;;;
;;;; 3. Repair.  Errors, warnings and failing tests (with the values they
;;;;    produced) go back to the model as the next turn of the conversation,
;;;;    until the tests pass or the attempts run out.

(in-package :mlx.llm)

;;; ------------------------------------------------------------------
;;; The lexer: a byte-at-a-time follower of the reader's structure

(defstruct (lisp-lexer (:constructor make-lisp-lexer ()) (:copier copy-lisp-lexer))
  (mode :start)          ; :start :normal :string :bar :line-comment :block-comment :done
  (depth 0 :type fixnum)  ; open parens
  (block-depth 0 :type fixnum)
  (pending nil))          ; :escape :hash :char, or in block comments :hash / :bar

(declaim (inline whitespace-byte-p))
(defun whitespace-byte-p (b) (member b '(32 9 10 13 12)))

(defun lex-byte (lx b)
  "Advance LX over byte B.  Returns NIL if B cannot continue a single
readable form (LX is then in an undefined state)."
  (let ((mode (lisp-lexer-mode lx)) (pending (lisp-lexer-pending lx)))
    (macrolet ((to (m) `(setf (lisp-lexer-mode lx) ,m))
               (pend (p) `(setf (lisp-lexer-pending lx) ,p)))
      (case mode
        ((:start :done)
         (cond ((whitespace-byte-p b) t)
               ((and (eq mode :start) (= b 40)) (to :normal) (setf (lisp-lexer-depth lx) 1) t)
               (t nil)))
        (:normal
         (cond
           ((eq pending :escape) (pend nil) t)
           ((eq pending :char) (pend nil) t)          ; #\x: x is literal
           (t
            (when (eq pending :hash)
              (pend nil)
              (case b
                (92 (pend :char) (return-from lex-byte t))                  ; #\
                (124 (to :block-comment) (setf (lisp-lexer-block-depth lx) 1) ; #|
                 (return-from lex-byte t))
                (40)                                                        ; #( vector
                (t (return-from lex-byte
                     ;; standard dispatch characters (and #nA / #n=); never
                     ;; #. (read-time evaluation) or undefined ones like #<
                     (and (or (<= 48 b 57) (find (code-char b) "'+-:xXbBoOcCpP*aAsS"))
                          t)))))
            (case b
              (40 (incf (lisp-lexer-depth lx)) t)
              (41 (when (zerop (decf (lisp-lexer-depth lx))) (to :done)) t)
              (34 (to :string) t)
              (124 (to :bar) t)
              (59 (to :line-comment) t)
              (92 (pend :escape) t)
              (35 (pend :hash) t)
              (t t)))))
        (:string
         (cond ((eq pending :escape) (pend nil) t)
               ((= b 92) (pend :escape) t)
               ((= b 34) (to :normal) t)
               (t t)))
        (:bar
         (cond ((eq pending :escape) (pend nil) t)
               ((= b 92) (pend :escape) t)
               ((= b 124) (to :normal) t)
               (t t)))
        (:line-comment
         (when (= b 10) (to :normal))
         t)
        (:block-comment
         (cond ((and (eq pending :hash) (= b 124))              ; nested #|
                (incf (lisp-lexer-block-depth lx)) (pend nil))
               ((and (eq pending :bar) (= b 35))                ; |#
                (pend nil)
                (when (zerop (decf (lisp-lexer-block-depth lx))) (to :normal)))
               (t (pend (case b (35 :hash) (124 :bar) (t nil)))))
         t)))))

(defun lex-octets (lx octets)
  "Advance LX over OCTETS; NIL if they break the form."
  (loop for b across octets always (lex-byte lx b)))

(defun lisp-form-complete-p (lx) (eq (lisp-lexer-mode lx) :done))

;;; ------------------------------------------------------------------
;;; Masks over the vocabulary

(defstruct (lisp-constraint (:constructor %make-lisp-constraint))
  octets        ; token id -> bytes
  classes       ; token id -> :ws :plain :structural :forbidden
  structural)   ; ids to simulate each step

(defparameter +structural-bytes+ '(40 41 34 124 59 92 35 46)
  "( ) \" | ; \\ # . -- bytes that can change the reader's structure.")

(defun make-lisp-constraint (tokenizer)
  "Classify every token of TOKENIZER once."
  (let* ((n (vocab-size tokenizer))
         (octets (make-array n))
         (classes (make-array n))
         (structural '()))
    (dotimes (id n)
      (let ((bytes (if (gethash id (slot-value tokenizer 'special-ids))
                       nil
                       (ignore-errors (token-octets tokenizer id)))))
        (setf (aref octets id) bytes
              (aref classes id)
              (cond ((or (null bytes) (zerop (length bytes))) :forbidden)
                    ((every #'whitespace-byte-p bytes) :ws)
                    ((some (lambda (b) (member b +structural-bytes+)) bytes)
                     (push id structural) :structural)
                    (t :plain)))))
    (%make-lisp-constraint :octets octets :classes classes :structural (nreverse structural))))

(defun lisp-mask (constraint lexer vocab)
  "An additive mask (0 or -inf) over VOCAB logits allowing exactly the
tokens that keep LEXER's text a prefix of one readable form."
  (let* ((classes (lisp-constraint-classes constraint))
         (octets (lisp-constraint-octets constraint))
         (outside (member (lisp-lexer-mode lexer) '(:start :done)))
         ;; after #, the next byte picks a dispatch function: check every token
         (after-hash (eq (lisp-lexer-pending lexer) :hash))
         (mask (make-array vocab :element-type 'single-float
                                 :initial-element sb-ext:single-float-negative-infinity)))
    (dotimes (id (min vocab (length classes)))
      (case (svref classes id)
        (:ws (setf (aref mask id) 0f0))
        ;; no plain token can change the structure, except after "#"
        (:plain (unless outside
                  (when (or (not after-hash) (lex-octets (copy-lisp-lexer lexer) (svref octets id)))
                    (setf (aref mask id) 0f0))))))
    (dolist (id (lisp-constraint-structural constraint))
      (when (and (< id vocab) (lex-octets (copy-lisp-lexer lexer) (svref octets id)))
        (setf (aref mask id) 0f0)))
    (mx:from-lisp mask)))

;;; ------------------------------------------------------------------
;;; Generating one form

(defvar *constraints* (make-hash-table :test 'eq :weakness :key)
  "Tokenizer -> its LISP-CONSTRAINT, built on first use.")

(defun tokenizer-constraint (tokenizer)
  (or (gethash tokenizer *constraints*)
      (setf (gethash tokenizer *constraints*) (make-lisp-constraint tokenizer))))

(defun closing-text (lexer)
  "Text that completes the open form LEXER is in: end any string, |symbol|
or comment, then close every open paren."
  (with-output-to-string (s)
    (case (lisp-lexer-mode lexer)
      (:string (write-char #\" s))
      (:bar (write-char #\| s))
      (:line-comment (terpri s))
      (:block-comment (dotimes (i (lisp-lexer-block-depth lexer)) (write-string "|#" s))))
    (when (lisp-lexer-pending lexer) (write-char #\Space s))
    (dotimes (i (lisp-lexer-depth lexer)) (write-char #\) s))))

(defvar *constraint-trace* nil
  "A stream, or NIL.  GENERATE-LISP-FORM reports there each token where the
reader constraint overrode the model's own choice.")

(defun generate-lisp-form (model prompt-ids &key (max-tokens 512) (sampler (make-sampler)))
  "Generate one Common Lisp form after PROMPT-IDS, decoding under the reader
constraint.  When the model tries to end its reply (its unconstrained choice
is a special token such as end-of-turn) with parens still open -- it lost
count -- the form is closed for it.  Returns (values text complete-p
closing): COMPLETE-P is false if MAX-TOKENS ran out first; CLOSING is the
text added to close the form, or NIL."
  (let* ((tokenizer (model-tokenizer model))
         (constraint (tokenizer-constraint tokenizer))
         (special (slot-value tokenizer 'special-ids))
         (lexer (make-lisp-lexer))
         (cache (make-cache model))
         (vocab nil)
         (ids '())
         (closing nil))
    ;; prefill all but the last prompt token, as GENERATE-TOKENS does
    (when (> (length prompt-ids) 1)
      (mx:with-scope ()
        (funcall model (mx:from-lisp (list (butlast prompt-ids)) :dtype :int32) cache)
        (mx:eval (loop for c in cache collect (kv-cache-keys c)))))
    (let ((input (last prompt-ids)))
      (loop repeat max-tokens
            do (multiple-value-bind (id wants-to-stop)
                   (mx:with-scope ()
                     (let* ((logits (mx:ref (funcall model (mx:from-lisp (list input) :dtype :int32) cache)
                                            t -1))
                            (vocab* (or vocab (setf vocab (mx:dim logits -1))))
                            (unconstrained (mx:item (mx:argmax logits :axis -1)))
                            (chosen (mx:item (funcall sampler (mx:add logits (lisp-mask constraint lexer vocab*))))))
                       (when (and *constraint-trace* (/= chosen unconstrained))
                         (format *constraint-trace* "~&;; [~(~A~) depth ~D] model wanted ~S, got ~S~%"
                                 (lisp-lexer-mode lexer) (lisp-lexer-depth lexer)
                                 (decode tokenizer (list unconstrained)) (decode tokenizer (list chosen))))
                       (values chosen
                               (or (gethash unconstrained special)
                                   (member unconstrained (eos-tokens tokenizer))))))
                 (when (and wants-to-stop (not (eq (lisp-lexer-mode lexer) :start)))
                   (setf closing (closing-text lexer))
                   (return))
                 (lex-octets lexer (svref (lisp-constraint-octets constraint) id))
                 (push id ids)
                 (setf input (list id))
                 (when (lisp-form-complete-p lexer) (return)))))
    (values (string-trim '(#\Space #\Tab #\Newline #\Return #\Page)
                         (concatenate 'string (decode tokenizer (reverse ids)) (or closing "")))
            (or (lisp-form-complete-p lexer) (and closing t))
            closing)))

;;; ------------------------------------------------------------------
;;; Parens from indentation
;;;
;;; Models indent Lisp well but lose count of closing parens.  When the two
;;; disagree the indentation is usually right, so, like Parinfer's indent
;;; mode, we can drop the closers at the end of each line and put back the
;;; ones the indentation implies: a line indented to column C closes every
;;; form opened at column C or beyond.

(defun classify-line (line mode depth)
  "Classify each character of LINE, starting in reader MODE (:code, :string,
:bar or :block, with DEPTH nested block comments).  Returns (values classes
mode depth): classes is a vector of :open, :close, :ws, :content or :comment."
  (let ((classes (make-array (length line))) (i 0) (n (length line)))
    (flet ((mark (class &optional (count 1))
             (loop repeat count while (< i n) do (setf (aref classes i) class) (incf i))))
      (loop while (< i n)
            do (let ((c (char line i)))
                 (ecase mode
                   (:code
                    (case c
                      (#\( (mark :open))
                      (#\) (mark :close))
                      ((#\Space #\Tab #\Page #\Return) (mark :ws))
                      (#\; (mark :comment n))
                      (#\\ (mark :content 2))
                      (#\" (setf mode :string) (mark :content))
                      (#\| (setf mode :bar) (mark :content))
                      (#\# (cond ((and (< (1+ i) n) (char= (char line (1+ i)) #\|))
                                  (setf mode :block depth 1) (mark :comment 2))
                                 ((and (< (1+ i) n) (char= (char line (1+ i)) #\\))
                                  (mark :content 3))
                                 (t (mark :content))))
                      (t (mark :content))))
                   ((:string :bar)
                    (cond ((char= c #\\) (mark :content 2))
                          ((char= c (if (eq mode :string) #\" #\|)) (setf mode :code) (mark :content))
                          (t (mark :content))))
                   (:block
                    (cond ((and (char= c #\|) (< (1+ i) n) (char= (char line (1+ i)) #\#))
                           (when (zerop (decf depth)) (setf mode :code))
                           (mark :comment 2))
                          ((and (char= c #\#) (< (1+ i) n) (char= (char line (1+ i)) #\|))
                           (incf depth) (mark :comment 2))
                          (t (mark :comment))))))))
    (values classes mode depth)))

(defun indentation-parens (code)
  "CODE with its closing parens re-derived from its indentation."
  (let ((lines (coerce (uiop:split-string code :separator '(#\Newline)) 'vector))
        (stack '())                     ; columns of the open parens
        (mode :code) (depth 0)
        (last nil) (insert 0))          ; the last code line, and where its closers go
    (flet ((close-to (column)
             (let ((n (loop while (and stack (>= (first stack) column)) do (pop stack) count t)))
               (when (and last (plusp n))
                 (let ((line (aref lines last)))
                   (setf (aref lines last) (concatenate 'string (subseq line 0 insert)
                                                        (make-string n :initial-element #\))
                                                        (subseq line insert)))
                   (incf insert n))))))
      (dotimes (k (length lines))
        (let ((line (aref lines k)) (start-mode mode))
          (multiple-value-bind (classes end-mode end-depth) (classify-line line mode depth)
            (setf mode end-mode depth end-depth)
            (let* ((indent (or (position-if-not (lambda (c) (member c '(#\Space #\Tab))) line) (length line)))
                   (content-end (let ((p (position-if (lambda (c) (member c '(:open :content))) classes
                                                      :from-end t)))
                                  (if p (1+ p) 0)))
                   (comment (position :comment classes :start content-end)))
              (when (and (eq start-mode :code) (plusp content-end))
                (close-to indent))
              (let ((head (make-array content-end :element-type 'character :fill-pointer 0)))
                (dotimes (i content-end)
                  (case (aref classes i)
                    (:open (push (fill-pointer head) stack) (vector-push (char line i) head))
                    (:close (when stack           ; an unmatched closer is dropped
                              (pop stack)
                              (vector-push (char line i) head)))
                    (t (vector-push (char line i) head))))
                (let ((head (coerce head 'simple-string)))
                  (setf (aref lines k)
                        (cond ((plusp content-end)
                               (if comment (concatenate 'string head " " (subseq line comment)) head))
                              (comment line)
                              ((find :close classes) nil) ; a line of closers only: gone
                              (t line)))
                  (when (plusp content-end)
                    (setf last k insert (length head)))))))))
      (close-to 0)
      (format nil "~{~A~^~%~}" (remove nil (coerce lines 'list))))))

(defun same-parens-p (a b)
  (string= (remove-if #'whitespace-char-p a) (remove-if #'whitespace-char-p b)))

(defun whitespace-char-p (c) (member c '(#\Space #\Tab #\Newline #\Return #\Page)))

;;; ------------------------------------------------------------------
;;; Evaluating code and tests

(defparameter *evaluator-source*
  "(lambda (code tests)
     ;; Runs in the evaluator (a child SBCL, or a thread of this one).
     ;; Returns a plist describing what happened.
     (let* ((package (make-package (gensym \"SANDBOX\") :use '(\"COMMON-LISP\")))
            (*package* package)
            (*read-eval* nil)
            (warnings '())
            (output (make-string-output-stream)))
       (flet ((unqualify (text)       ; drop the sandbox's package prefix
                (let ((prefix (concatenate 'string (package-name package) \"::\")))
                  (loop for p = (search prefix text) while p
                        do (setf text (concatenate 'string (subseq text 0 p)
                                                   (subseq text (+ p (length prefix))))))
                  text))
              (show (x) (let ((*package* package) (*print-length* 40) (*print-level* 8) (*print-circle* t))
                         (prin1-to-string x))))
         (unwind-protect
              (handler-case
                  (handler-bind ((warning (lambda (w)
                                            (push (unqualify (princ-to-string w)) warnings)
                                            (muffle-warning w))))
                    (let* ((*standard-output* output)
                           (forms (with-input-from-string (in code)
                                    (loop for f = (read in nil in) until (eq f in) collect f)))
                           (value (let (v) (dolist (f forms v) (setf v (eval f)))))
                           (failures
                             (loop for text in tests
                                   for test = (read-from-string text)
                                   unless (eval test)
                                     collect (if (and (consp test) (= (length test) 3)
                                                      (member (car test) '(= eql equal equalp string= char=)))
                                                 (format nil \"~A: ~A returned ~A, expected ~A\" (show test)
                                                         (show (second test)) (show (eval (second test)))
                                                         (show (eval (third test))))
                                                 (format nil \"~A returned false\" (show test))))))
                      (list :ok (null failures) :value (show value)
                            :output (get-output-stream-string output)
                            :failures failures :warnings (reverse warnings))))
                (serious-condition (e)
                  (list :ok nil :error (unqualify (format nil \"~A: ~A\" (type-of e) e))
                        :output (get-output-stream-string output) :warnings (reverse warnings))))
           (delete-package package)))))"
  "The evaluator, as CL-USER source text: it runs in a child SBCL (which has
none of our packages) or, compiled, in a thread of this image.  Given CODE
and TESTS strings it returns a plist describing what happened.")

(defun evaluator-function ()
  (compile nil (let ((*package* (find-package :cl-user))) (read-from-string *evaluator-source*))))

(defun evaluate-in-process (code tests timeout)
  (let* ((fn (evaluator-function))
         (result nil)
         (thread (sb-thread:make-thread (lambda () (setf result (funcall fn code tests)))
                                        :name "lisp evaluator")))
    (if (eq (sb-thread:join-thread thread :default :timeout :timeout timeout) :timeout)
        (progn (sb-thread:terminate-thread thread)
               (list :ok nil :error (format nil "timed out after ~A seconds" timeout)))
        result)))

(defun sbcl-program ()
  "The SBCL to run evaluations in: $MLX_CL_SBCL, else this runtime if it
is sbcl (a saved application such as mlx-cl is not), else sbcl on PATH."
  (let ((env (uiop:getenv "MLX_CL_SBCL"))
        (runtime (and sb-ext:*runtime-pathname* (pathname sb-ext:*runtime-pathname*))))
    (cond ((and env (plusp (length env))) env)
          ((and runtime (equal (pathname-name runtime) "sbcl")) (uiop:native-namestring runtime))
          (t "sbcl"))))

(defun clip (string &optional (limit 600))
  "STRING, cut to LIMIT characters: errors and backtraces can be huge, and
they go into the model's prompt."
  (if (> (length string) limit)
      (format nil "~A~%... [~:D more characters]" (subseq string 0 limit) (- (length string) limit))
      string))

(defun evaluate-in-child (code tests timeout)
  "Run the evaluator in a fresh SBCL.  Its result is written to a file of
its own, so nothing the code prints can garble it."
  (uiop:with-temporary-file (:pathname script :type "lisp")
    (uiop:with-temporary-file (:pathname result-file :type "sexp")
      (uiop:with-temporary-file (:pathname error-file :type "txt")
        (with-open-file (out script :direction :output :if-exists :supersede :external-format :utf-8)
          (with-standard-io-syntax
            (let ((*print-readably* nil) (*package* (find-package :cl-user)))
              (format out "(let ((result (funcall ~A ~S '~S)))
    (with-open-file (out ~S :direction :output :if-exists :supersede :external-format :utf-8)
      (with-standard-io-syntax (let ((*print-readably* nil)) (prin1 result out)))))~%"
                      *evaluator-source* code tests (uiop:native-namestring result-file)))))
        (let* ((process (uiop:launch-program (list (sbcl-program)
                                                   "--noinform" "--non-interactive" "--no-sysinit" "--no-userinit"
                                                   "--load" (uiop:native-namestring script))
                                             :output nil :error-output error-file :if-error-output-exists :supersede))
               (deadline (+ (get-internal-real-time) (* timeout internal-time-units-per-second))))
          (loop while (and (uiop:process-alive-p process) (< (get-internal-real-time) deadline))
                do (sleep 0.02))
          (cond ((uiop:process-alive-p process)
                 (uiop:terminate-process process :urgent t)
                 (uiop:wait-process process)
                 (list :ok nil :error (format nil "timed out after ~A seconds" timeout)))
                (t
                 (or (ignore-errors
                      (with-open-file (in result-file :external-format :utf-8)
                        (let ((*read-eval* nil)) (read in))))
                     (list :ok nil
                           :error (format nil "the evaluator exited without a result:~%~A"
                                          (clip (uiop:read-file-string error-file))))))))))))

(defun evaluate-lisp (code &key tests (timeout 10) (isolation :process))
  "Read and evaluate CODE (a string of forms) in a fresh package using CL,
then each test (strings, each a form that must return true).  ISOLATION
:PROCESS (the default) runs it in a child SBCL; :IN-PROCESS runs it in a
thread of this image.  Returns a plist: :OK, :VALUE, :OUTPUT, :FAILURES,
:WARNINGS or :ERROR."
  (ecase isolation
    (:process (evaluate-in-child code tests timeout))
    (:in-process (evaluate-in-process code tests timeout))))

;;; ------------------------------------------------------------------
;;; The write / run / repair loop

(defparameter *lisp-system-prompt*
  "You are an expert Common Lisp programmer. Reply with exactly one Common Lisp form and nothing else: no explanation and no Markdown. To define several things, wrap them in PROGN. Use only standard Common Lisp: no libraries are available (no Quicklisp, Alexandria, split-sequence or cl-ppcre), so write any helpers yourself. Do not include tests or example calls: the tests are run separately.")

(defun feedback (result)
  "The message telling the model what went wrong with its code."
  (with-output-to-string (s)
    (cond ((getf result :error)
           (format s "Evaluating your code signalled an error:~%~A~%" (clip (getf result :error)))
           (when (search "UNDEFINED-FUNCTION" (getf result :error))
             (format s "That function is not part of Common Lisp, and no libraries are loaded: ~
                        define it yourself, in the same PROGN.~%"))
           (when (search "PACKAGE-ERROR" (getf result :error))
             (format s "Only the COMMON-LISP package exists: no libraries can be loaded. ~
                        Write the helper functions you need yourself.~%")))
          ((getf result :failures)
           (format s "Your code ran, but these tests failed:~%~{- ~A~%~}"
                   (mapcar #'clip (getf result :failures)))))
    (when (getf result :warnings)
      (format s "Compiler warnings:~%~{- ~A~%~}"
              (mapcar (lambda (w) (clip w 300))
                      (subseq (getf result :warnings) 0 (min 5 (length (getf result :warnings)))))))
    (format s "Reply with a corrected version, as one form.")))

(defun check-candidates (code tests timeout isolation)
  "Evaluate CODE and, when its indentation implies different parens, the
repaired version too (first).  Returns (values code result repaired-p) for
the version to keep: the first that passes, else the repaired one."
  (let ((repaired (indentation-parens code)))
    (if (same-parens-p repaired code)
        (values code (evaluate-lisp code :tests tests :timeout timeout :isolation isolation) nil)
        (let ((result (evaluate-lisp repaired :tests tests :timeout timeout :isolation isolation)))
          (if (getf result :ok)
              (values repaired result t)
              (let ((original (evaluate-lisp code :tests tests :timeout timeout :isolation isolation)))
                (if (getf original :ok)
                    (values code original nil)
                    (values repaired result t))))))))

(defun write-lisp (model task &key tests (attempts 4) (isolation :process) (timeout 10)
                                   (max-tokens 768) (temperature 0.0) (retry-temperature 0.7)
                                   (stream *standard-output*))
  "Ask MODEL to write Common Lisp for TASK (a string); TESTS are strings of
forms that must return true.  Generates one reader-valid form, re-derives
its closing parens from its indentation when the two disagree, evaluates it
with the tests (see EVALUATE-LISP), and feeds any error, warnings or test
failures back, up to ATTEMPTS times.  The first attempt samples at
TEMPERATURE, later ones at RETRY-TEMPERATURE (or TEMPERATURE if higher),
and an attempt that repeats an earlier one is sampled again.  Progress goes to STREAM.  Returns
(values code success-p result attempts-used)."
  (let* ((tokenizer (model-tokenizer model))
         (sampler (make-sampler :temperature temperature))
         ;; greedy retries tend to resubmit the same code
         (retry-sampler (make-sampler :temperature (max temperature retry-temperature)))
         (messages (list (cons "system" *lisp-system-prompt*)
                         (cons "user" (format nil "~A~@[~%~%It must pass these tests:~%~{~A~%~}~]" task tests))))
         (seen '()) (code nil) (result nil))
    (loop for attempt from 1 to attempts
          do (let ((prompt (encode tokenizer (apply-chat-template tokenizer messages :thinking nil))))
               (multiple-value-bind (text complete closed)
                   (generate-lisp-form model prompt :max-tokens max-tokens
                                                   :sampler (if (= attempt 1) sampler retry-sampler))
                 (when (member text seen :test #'string=)
                   (multiple-value-setq (text complete closed)
                     (generate-lisp-form model prompt :max-tokens max-tokens :sampler retry-sampler)))
                 (push text seen)
                 (let ((repaired nil))
                   (if complete
                       (multiple-value-setq (code result repaired)
                         (check-candidates text tests timeout isolation))
                       (setf code text
                             result (list :ok nil :error (format nil "the form was not finished within ~D tokens"
                                                                 max-tokens))))
                   (when stream
                     (format stream "~&;; attempt ~D~@[, closing ~D paren~:P the model left open~]~:[~;, parens repaired from indentation~]~%~A~%;; => ~:[~A~;ok~@[, value ~A~]~]~%"
                             attempt (and closed (count #\) closed)) repaired
                             code
                             (getf result :ok)
                             (if (getf result :ok)
                                 (getf result :value)
                                 (clip (substitute #\Space #\Newline
                                                   (or (getf result :error)
                                                       (format nil "~D failed test~:P" (length (getf result :failures)))))
                                       200)))))
                 (when (getf result :ok)
                   (return-from write-lisp (values code t result attempt)))
                 (setf messages (append messages (list (cons "assistant" code)
                                                       (cons "user" (feedback result))))))))
    (values code nil result attempts)))
