;;;; llm/tokenizer.lisp -- BPE tokenizers from Hugging Face tokenizer.json
;;;;
;;;; Two families are implemented:
;;;;   byte-level BPE  -- GPT-2, SmolLM, Llama 3, Qwen2: text is split by a
;;;;                      pre-tokenizer regex and mapped byte by byte to
;;;;                      printable characters before merging;
;;;;   SentencePiece BPE -- Gemma, Phi-3, Llama 2, Mistral: spaces become
;;;;                      U+2581 (▁), merging works on characters, and
;;;;                      characters missing from the vocabulary fall back to
;;;;                      <0xNN> byte tokens.
;;;; Encoding follows the tokenizers library step by step: split out added
;;;; tokens (honouring lstrip/rstrip), normalize each remaining segment,
;;;; pre-tokenize, then merge pairs by rank.  Regexes are hand-written
;;;; scanners with the exact regex semantics, so no regex engine is needed.
;;;; Unknown configurations are rejected rather than approximated: a
;;;; slightly wrong tokenizer silently degrades a model.

(in-package :mlx.llm)

(defclass tokenizer ()
  ((vocab :initarg :vocab :documentation "token string -> id")
   (id->token :initarg :id->token :documentation "vector id -> token string")
   (ranks :initarg :ranks :documentation "merge key (see MERGE-KEY) -> rank")
   (special :initarg :special :initform '()
            :documentation "added tokens as ADDED structs, longest first")
   (special-index :initform nil :documentation "first char -> added tokens starting with it")
   (special-ids :initarg :special-ids :documentation "hash id -> t for special tokens")
   (pre-tokenizers :initarg :pre-tokenizers :initform '()
                   :documentation "list of steps: :digits, :gpt2, (:llama3 n), (:split s behavior),
(:metaspace replacement prepend-scheme split)")
   (normalizer :initarg :normalizer :initform #'identity)
   (byte-level :initarg :byte-level :initform t
               :documentation "true: byte-level BPE; false: SentencePiece-style BPE")
   (byte-fallback :initarg :byte-fallback :initform nil)
   (unk-id :initarg :unk-id :initform nil)
   (ignore-merges :initarg :ignore-merges :initform nil
                  :documentation "a pre-token already in the vocabulary is not merged")
   (strip-leading-space :initarg :strip-leading-space :initform nil
                        :documentation "decoder strips one leading space (SentencePiece Strip)")
   (cache :initform (make-hash-table :test 'equal))
   (chat-template :initarg :chat-template :initform nil :reader tokenizer-chat-template)
   (bos-token :initarg :bos-token :initform nil :reader bos-token)
   (bos-string :initarg :bos-string :initform nil :reader bos-string)
   (eos-string :initarg :eos-string :initform nil :reader eos-string)
   (eos-tokens :initarg :eos-tokens :initform '() :accessor eos-tokens
               :documentation "ids that end generation")))

(defstruct (added (:constructor make-added (content id &key lstrip rstrip)))
  content id lstrip rstrip)

(defmethod shared-initialize :after ((tk tokenizer) slots &key special)
  (declare (ignore slots))
  ;; accept (content . id) conses as well as ADDED structs
  (when special
    (setf (slot-value tk 'special)
          (sort (mapcar (lambda (s) (if (consp s) (make-added (car s) (cdr s)) s)) special)
                #'> :key (lambda (a) (length (added-content a))))))
  (let ((index (make-hash-table)))
    (dolist (a (reverse (slot-value tk 'special)))
      (when (plusp (length (added-content a)))
        (push a (gethash (char (added-content a) 0) index))))
    ;; longest first within each bucket
    (maphash (lambda (k v) (setf (gethash k index) (sort v #'> :key (lambda (a) (length (added-content a))))))
             index)
    (setf (slot-value tk 'special-index) index)))

(defmethod print-object ((tk tokenizer) stream)
  (print-unreadable-object (tk stream :type t)
    (format stream "~:D tokens~:[, SentencePiece~;~]" (hash-table-count (slot-value tk 'vocab))
            (slot-value tk 'byte-level))))

(defun vocab-size (tokenizer) (length (slot-value tokenizer 'id->token)))

(defun token-id (tokenizer token)
  "The id of the token string TOKEN, or NIL."
  (values (gethash token (slot-value tokenizer 'vocab))))

(defun merge-key (left right)
  "Hash key for the merge of the token strings LEFT and RIGHT."
  (concatenate 'string left (string (code-char 0)) right))

;;; ------------------------------------------------------------------
;;; Byte <-> unicode mapping of byte-level BPE (GPT-2's bytes_to_unicode)

(defparameter *byte->char*
  (let ((table (make-array 256)) (n 0))
    (dotimes (b 256)
      (setf (aref table b)
            (if (or (<= 33 b 126) (<= 161 b 172) (<= 174 b 255))
                (code-char b)
                (prog1 (code-char (+ 256 n)) (incf n)))))
    table))

(defparameter *char->byte*
  (let ((h (make-hash-table)))
    (dotimes (b 256 h) (setf (gethash (aref *byte->char* b) h) b))))

(defun bytes->bpe-string (octets)
  (map 'string (lambda (b) (aref *byte->char* b)) octets))

(defconstant +metaspace+ (code-char #x2581) "SentencePiece's space marker, ▁.")

;;; ------------------------------------------------------------------
;;; Character classes (Unicode general categories, as \p{L} and \p{N})

(declaim (inline letterp numberp* spacep newlinep))
(defun letterp (c) (char= #\L (char (symbol-name (sb-unicode:general-category c)) 0)))
(defun numberp* (c) (char= #\N (char (symbol-name (sb-unicode:general-category c)) 0)))
(defun spacep (c) (sb-unicode:whitespace-p c))
(defun newlinep (c) (or (char= c #\Return) (char= c #\Newline)))
(defun otherp (c) (not (or (spacep c) (letterp c) (numberp* c))))

;;; ------------------------------------------------------------------
;;; Pre-tokenizer scanners.  Each returns the end of the match at START.

(defun run-end (s start pred)
  (or (position-if-not pred s :start start) (length s)))

(defun contraction-end (s i &key case-insensitive)
  "End of 's 't 're 've 'm 'll 'd at I, or NIL."
  (when (and (< i (length s)) (char= (char s i) #\'))
    (dolist (suffix '("s" "t" "re" "ve" "m" "ll" "d"))
      (let ((end (+ i 1 (length suffix))))
        (when (and (<= end (length s))
                   (funcall (if case-insensitive #'string-equal #'string=)
                            suffix s :start2 (1+ i) :end2 end))
          (return end))))))

(defun whitespace-end (s i)
  "\\s+(?!\\S) then \\s+ at I (I is whitespace)."
  (let ((end (run-end s i #'spacep)))
    (cond ((= end (length s)) end)            ; run reaches the end
          ((> (- end i) 1) (1- end))          ; leave one space for the next word
          (t end))))                          ; \s+ fallback: the single char

(defun gpt2-match-end (s i)
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
  (let* ((n (length s))
         (c (char s i))
         (j (if (and (char= c #\Space) (< (1+ i) n)) (1+ i) i))
         (d (char s j)))
    (or (contraction-end s i)
        (and (letterp d) (run-end s j #'letterp))
        (and (numberp* d) (run-end s j #'numberp*))
        (and (otherp d) (run-end s j #'otherp))
        (whitespace-end s i))))

(defun llama3-match-end (s i max-digits)
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,MAX}
| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
  (let* ((n (length s))
         (c (char s i)))
    (or (contraction-end s i :case-insensitive t)
        ;; [^\r\n\p{L}\p{N}]?\p{L}+
        (cond ((letterp c) (run-end s i #'letterp))
              ((and (not (newlinep c)) (not (numberp* c)) (< (1+ i) n) (letterp (char s (1+ i))))
               (run-end s (1+ i) #'letterp)))
        ;; \p{N}{1,max}
        (and (numberp* c) (min (run-end s i #'numberp*) (+ i max-digits)))
        ;;  ?[^\s\p{L}\p{N}]+[\r\n]*
        (let ((j (if (and (char= c #\Space) (< (1+ i) n)) (1+ i) i)))
          (and (otherp (char s j))
               (run-end s (run-end s j #'otherp) #'newlinep)))
        ;; \s*[\r\n]+ : up to the last newline in the whitespace run
        (let* ((end (run-end s i #'spacep))
               (last-newline (position-if #'newlinep s :start i :end end :from-end t)))
          (and last-newline (1+ last-newline)))
        (whitespace-end s i))))

(defun scan (s matcher)
  "Split S into the successive matches of MATCHER."
  (loop with i = 0
        while (< i (length s))
        collect (let ((end (funcall matcher s i)))
                  (prog1 (subseq s i end) (setf i end)))))

(defun split-digits (s)
  "The Digits pre-tokenizer (individual digits): isolate every numeric char."
  (let ((out '()) (start 0))
    (loop for i from 0 below (length s)
          when (numberp* (char s i))
            do (when (> i start) (push (subseq s start i) out))
               (push (string (char s i)) out)
               (setf start (1+ i)))
    (when (< start (length s)) (push (subseq s start) out))
    (nreverse out)))

(defun split-on-string (s pattern behavior)
  "The Split pre-tokenizer for a literal PATTERN with the given BEHAVIOR."
  (let ((n (length pattern)) (pieces '()) (start 0))
    (loop for pos = (search pattern s :start2 start)
          while pos
          do (ecase behavior
               (:merged-with-previous (push (subseq s start (+ pos n)) pieces) (setf start (+ pos n)))
               (:merged-with-next (push (subseq s start pos) pieces)
                (push (subseq s pos (+ pos n)) pieces) (setf start (+ pos n))
                ;; the pattern belongs to the text after it
                (setf pieces (cdr pieces) start pos)
                (let ((next (search pattern s :start2 (+ pos n))))
                  (push (subseq s pos (or next (length s))) pieces)
                  (setf start (or next (length s)))))
               (:isolated (push (subseq s start pos) pieces) (push pattern pieces) (setf start (+ pos n)))
               (:removed (push (subseq s start pos) pieces) (setf start (+ pos n)))))
    (push (subseq s start) pieces)
    (remove "" (nreverse pieces) :test #'string=)))

(defun metaspace (s replacement prepend-scheme split first)
  "The Metaspace pre-tokenizer: spaces become REPLACEMENT, optionally
prepended, optionally split before each REPLACEMENT."
  (let* ((s (substitute replacement #\Space s))
         (s (if (and (plusp (length s)) (char/= (char s 0) replacement)
                     (or (eq prepend-scheme :always) (and (eq prepend-scheme :first) first)))
                (concatenate 'string (string replacement) s)
                s)))
    (if split
        (loop with start = 0
              for pos = (position replacement s :start (1+ start))
              collect (subseq s start pos)
              while pos do (setf start pos))
        (list s))))

(defun pre-tokenize (tokenizer text &key (first t))
  "Split TEXT into the pieces merged separately.  FIRST says whether TEXT
starts the input (for Metaspace's :first scheme)."
  (let ((pieces (list text)))
    (dolist (step (slot-value tokenizer 'pre-tokenizers) pieces)
      (setf pieces
            (mapcan (lambda (piece)
                      (ecase (if (consp step) (first step) step)
                        (:digits (split-digits piece))
                        (:gpt2 (scan piece #'gpt2-match-end))
                        (:llama3 (let ((max (second step)))
                                   (scan piece (lambda (s i) (llama3-match-end s i max)))))
                        (:split (split-on-string piece (second step) (third step)))
                        (:metaspace (destructuring-bind (replacement scheme split) (rest step)
                                      (metaspace piece replacement scheme split first)))))
                    pieces)))))

;;; ------------------------------------------------------------------
;;; BPE merging
;;;
;;; Symbols live in a doubly linked list over vectors; candidate pairs sit in
;;; a binary heap ordered by (rank, position) and are checked for staleness
;;; when popped.  O(n log n), so SentencePiece segments of whole paragraphs
;;; (no pre-tokenizer) stay fast.

(declaim (inline pair<))
(defun pair< (a b)
  ;; entries are (rank position left-string right-string)
  (or (< (first a) (first b)) (and (= (first a) (first b)) (< (second a) (second b)))))

(defun heap-push (heap item)
  (vector-push-extend item heap)
  (let ((i (1- (length heap))))
    (loop while (plusp i)
          do (let ((parent (floor (1- i) 2)))
               (if (pair< (aref heap i) (aref heap parent))
                   (progn (rotatef (aref heap i) (aref heap parent)) (setf i parent))
                   (return))))))

(defun heap-pop (heap)
  (let ((top (aref heap 0)) (last (vector-pop heap)))
    (when (plusp (length heap))
      (setf (aref heap 0) last)
      (let ((i 0) (n (length heap)))
        (loop (let* ((l (1+ (* 2 i))) (r (1+ l)) (smallest i))
                (when (and (< l n) (pair< (aref heap l) (aref heap smallest))) (setf smallest l))
                (when (and (< r n) (pair< (aref heap r) (aref heap smallest))) (setf smallest r))
                (when (= smallest i) (return))
                (rotatef (aref heap i) (aref heap smallest))
                (setf i smallest)))))
    top))

(defun bpe-merge (ranks symbols)
  "Merge the list of token strings SYMBOLS by RANKS; returns the result list."
  (let* ((n (length symbols))
         (syms (coerce symbols 'simple-vector))
         (next (make-array n)) (prev (make-array n))
         (heap (make-array 16 :adjustable t :fill-pointer 0)))
    (dotimes (i n)
      (setf (aref next i) (if (< (1+ i) n) (1+ i) nil)
            (aref prev i) (if (plusp i) (1- i) nil)))
    (flet ((consider (i)
             (let ((j (and i (aref next i))))
               (when j
                 (let ((rank (gethash (merge-key (aref syms i) (aref syms j)) ranks)))
                   (when rank (heap-push heap (list rank i (aref syms i) (aref syms j)))))))))
      (dotimes (i (1- n)) (consider i))
      (loop while (plusp (length heap))
            do (destructuring-bind (rank i left right) (heap-pop heap)
                 (declare (ignore rank))
                 (let ((j (aref next i)))
                   ;; stale if either side has changed since the entry was pushed
                   (when (and (aref syms i) j (equal (aref syms i) left) (equal (aref syms j) right))
                     (setf (aref syms i) (concatenate 'string left right)
                           (aref syms j) nil
                           (aref next i) (aref next j))
                     (when (aref next j) (setf (aref prev (aref next j)) i))
                     (consider (aref prev i))
                     (consider i))))))
    (loop for i = 0 then (aref next i) while i collect (aref syms i))))

(defun initial-symbols (tokenizer word)
  "The symbols WORD starts from: byte-mapped characters (byte-level), or
characters with <0xNN> byte fallback for ones not in the vocabulary."
  (if (slot-value tokenizer 'byte-level)
      (map 'list #'string (bytes->bpe-string (sb-ext:string-to-octets word :external-format :utf-8)))
      (let ((vocab (slot-value tokenizer 'vocab)))
        (loop for c across word
              for s = (string c)
              nconc (if (or (gethash s vocab) (not (slot-value tokenizer 'byte-fallback)))
                        (list s)
                        (map 'list (lambda (b) (format nil "<0x~2,'0X>" b))
                             (sb-ext:string-to-octets s :external-format :utf-8)))))))

(defun bpe (tokenizer word)
  "Token ids for the pre-tokenized WORD."
  (let ((cache (slot-value tokenizer 'cache))
        (vocab (slot-value tokenizer 'vocab)))
    (or (and (< (length word) 64) (gethash word cache))
        (let* ((mapped (if (slot-value tokenizer 'byte-level)
                           (bytes->bpe-string (sb-ext:string-to-octets word :external-format :utf-8))
                           word))
               (ids (if (and (slot-value tokenizer 'ignore-merges) (gethash mapped vocab))
                        (list (gethash mapped vocab))
                        (loop for token in (bpe-merge (slot-value tokenizer 'ranks)
                                                      (initial-symbols tokenizer word))
                              collect (or (gethash token vocab)
                                          (slot-value tokenizer 'unk-id)
                                          (error "Token ~S is not in the vocabulary." token))))))
          (when (< (length word) 64) (setf (gethash word cache) ids))
          ids))))

(defun encode-plain (tokenizer text &key (first t))
  (loop for piece in (pre-tokenize tokenizer (funcall (slot-value tokenizer 'normalizer) text)
                                   :first first)
        nconc (copy-list (bpe tokenizer piece))))

(defun split-special (tokenizer text)
  "Split TEXT into strings and added-token ids, honouring each added
token's lstrip/rstrip (which absorb neighbouring whitespace)."
  (let ((index (slot-value tokenizer 'special-index))
        (out '()) (start 0) (i 0) (n (length text)))
    (flet ((emit-text (end) (when (> end start) (push (subseq text start end) out))))
      (loop while (< i n)
            do (let ((hit (find-if (lambda (a) (let* ((c (added-content a)) (end (+ i (length c))))
                                                 (and (<= end n) (string= c text :start2 i :end2 end))))
                                   (gethash (char text i) index))))
                 (if hit
                     (let ((text-end (if (added-lstrip hit)
                                         (let ((e i))
                                           (loop while (and (> e start) (spacep (char text (1- e))))
                                                 do (decf e))
                                           e)
                                         i)))
                       (emit-text text-end)
                       (push (added-id hit) out)
                       (setf i (+ i (length (added-content hit))))
                       (when (added-rstrip hit)
                         (loop while (and (< i n) (spacep (char text i))) do (incf i)))
                       (setf start i))
                     (incf i))))
      (emit-text n))
    (nreverse out)))

(defun encode (tokenizer text &key add-bos)
  "Token ids for TEXT.  Added tokens written in TEXT (e.g. <|im_start|>)
become their ids.  ADD-BOS prepends the beginning-of-sequence token."
  (append (and add-bos (bos-token tokenizer) (list (bos-token tokenizer)))
          (loop for part in (split-special tokenizer text)
                for first = t then nil
                nconc (if (integerp part)
                          (list part)
                          (encode-plain tokenizer part :first first)))))

;;; ------------------------------------------------------------------
;;; Decoding

(defun byte-fallback-token-p (token)
  (and (= (length token) 6) (string= "<0x" token :end2 3) (char= (char token 5) #\>)))

(defun token-octets (tokenizer id)
  (let ((token (aref (slot-value tokenizer 'id->token) id)))
    (cond ((gethash id (slot-value tokenizer 'special-ids))
           (sb-ext:string-to-octets token :external-format :utf-8))
          ((slot-value tokenizer 'byte-level)
           (map '(vector (unsigned-byte 8)) (lambda (c) (gethash c *char->byte* 63)) token))
          ((byte-fallback-token-p token)
           (vector (parse-integer token :start 3 :end 5 :radix 16)))
          (t (sb-ext:string-to-octets (substitute #\Space +metaspace+ token) :external-format :utf-8)))))

(defun decode (tokenizer ids &key skip-special)
  "The text of the token IDS."
  (let ((octets (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
    (dolist (id ids)
      (unless (and skip-special (gethash id (slot-value tokenizer 'special-ids)))
        (loop for b across (token-octets tokenizer id) do (vector-push-extend b octets))))
    (let ((text (sb-ext:octets-to-string (coerce octets '(vector (unsigned-byte 8)))
                                         :external-format '(:utf-8 :replacement #\?))))
      (if (and (slot-value tokenizer 'strip-leading-space) (plusp (length text))
               (char= (char text 0) #\Space))
          (subseq text 1)
          text))))

(defstruct (stream-decoder (:constructor make-stream-decoder (tokenizer)))
  tokenizer
  (pending (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0))
  (started nil))

(defun utf8-complete-length (octets)
  "Length of the longest prefix of OCTETS that ends on a character boundary."
  (let ((n (length octets)))
    ;; look back at most 3 bytes for an incomplete trailing sequence
    (loop for back from 1 to (min 4 n)
          for b = (aref octets (- n back))
          do (cond ((< b #x80) (return n))                ; ASCII: complete
                   ((>= b #xC0)                           ; lead byte
                    (let ((need (cond ((>= b #xF0) 4) ((>= b #xE0) 3) (t 2))))
                      (return (if (>= back need) n (- n back))))))
          finally (return n))))

(defun decode-step (decoder id)
  "Feed token ID to DECODER; return the newly completed text (maybe \"\")."
  (let ((pending (stream-decoder-pending decoder))
        (tokenizer (stream-decoder-tokenizer decoder)))
    (loop for b across (token-octets tokenizer id)
          do (vector-push-extend b pending))
    (let* ((complete (utf8-complete-length pending))
           (text (sb-ext:octets-to-string (coerce (subseq pending 0 complete) '(vector (unsigned-byte 8)))
                                          :external-format '(:utf-8 :replacement #\?)))
           (rest (subseq pending complete)))
      (setf (fill-pointer pending) 0)
      (loop for b across rest do (vector-push-extend b pending))
      ;; the SentencePiece Strip decoder removes one space from the start
      (when (and (plusp (length text)) (not (stream-decoder-started decoder)))
        (setf (stream-decoder-started decoder) t)
        (when (and (slot-value tokenizer 'strip-leading-space) (char= (char text 0) #\Space))
          (setf text (subseq text 1))))
      text)))

;;; ------------------------------------------------------------------
;;; Loading tokenizer.json

(defparameter *gpt2-pattern*
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

(defparameter *llama3-patterns*
  '(("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" . 3)
    ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" . 1))
  "Split patterns used by Llama 3 (digits in runs of up to 3) and Qwen2
(single digits).")

(defun json-get (object &rest keys)
  "Follow KEYS through nested JSON objects.  JSON null reads as NIL."
  (let ((x object))
    (dolist (k keys (if (eq x 'null) nil x))
      (setf x (and (hash-table-p x) (gethash k x))))))

(defun string-pattern (spec)
  "The literal string of a {\"String\": ...} pattern, or NIL."
  (json-get spec "pattern" "String"))

(defun parse-normalizer (spec)
  "A function normalizing text for the tokenizer.json normalizer SPEC."
  (if (null spec)
      #'identity
      (let ((type (json-get spec "type")))
        (cond ((string= type "NFC") (lambda (s) (sb-unicode:normalize-string s :nfc)))
              ((string= type "NFKC") (lambda (s) (sb-unicode:normalize-string s :nfkc)))
              ((string= type "Prepend")
               (let ((prefix (json-get spec "prepend")))
                 (lambda (s) (if (plusp (length s)) (concatenate 'string prefix s) s))))
              ((and (string= type "Replace") (string-pattern spec))
               (let ((from (string-pattern spec)) (to (json-get spec "content")))
                 (lambda (s) (replace-all s from to))))
              ((string= type "Sequence")
               (let ((steps (map 'list #'parse-normalizer (json-get spec "normalizers"))))
                 (lambda (s) (reduce (lambda (acc f) (funcall f acc)) steps :initial-value s))))
              (t (error "Unsupported normalizer ~S." type))))))

(defun replace-all (s from to)
  (with-output-to-string (out)
    (loop with start = 0
          for pos = (search from s :start2 start)
          do (write-string s out :start start :end (or pos (length s)))
             (when pos (write-string to out))
          while pos do (setf start (+ pos (length from))))))

(defun keyword-of (name)
  "\"MergedWithPrevious\" -> :MERGED-WITH-PREVIOUS"
  (intern (string-upcase (with-output-to-string (s)
                           (loop for c across name for i from 0
                                 do (when (and (plusp i) (upper-case-p c)) (write-char #\- s))
                                    (write-char c s))))
          :keyword))

(defun parse-pre-tokenizer (spec)
  "List of pre-tokenizer steps for the tokenizer.json SPEC."
  (when spec
    (let ((type (json-get spec "type")))
      (cond
        ((string= type "Sequence")
         (loop for sub across (json-get spec "pretokenizers") nconc (parse-pre-tokenizer sub)))
        ((string= type "Digits")
         (unless (json-get spec "individual_digits")
           (error "Unsupported pre-tokenizer: Digits without individual_digits."))
         (list :digits))
        ((string= type "ByteLevel")
         (when (json-get spec "add_prefix_space")
           (error "Unsupported pre-tokenizer: ByteLevel with add_prefix_space."))
         (if (json-get spec "use_regex") (list :gpt2) '()))
        ((string= type "Metaspace")
         (list (list :metaspace (char (json-get spec "replacement") 0)
                     (keyword-of (or (json-get spec "prepend_scheme") "always"))
                     (json-get spec "split"))))
        ((and (string= type "Split") (string-pattern spec))
         (when (json-get spec "invert") (error "Unsupported pre-tokenizer: inverted Split."))
         (list (list :split (string-pattern spec) (keyword-of (json-get spec "behavior")))))
        ((string= type "Split")
         (let* ((pattern (json-get spec "pattern" "Regex"))
                (known (assoc pattern *llama3-patterns* :test #'equal)))
           (cond ((equal pattern *gpt2-pattern*) (list :gpt2))
                 (known (list (list :llama3 (cdr known))))
                 (t (error "Unsupported Split pre-tokenizer pattern: ~S" pattern)))))
        (t (error "Unsupported pre-tokenizer type ~S." type))))))

(defun decoder-types (spec)
  (when spec
    (if (equal (json-get spec "type") "Sequence")
        (loop for d across (json-get spec "decoders") nconc (decoder-types d))
        (list spec))))

(defun load-tokenizer (directory)
  "Load the tokenizer.json (and tokenizer_config.json) in DIRECTORY."
  (let* ((dir (uiop:ensure-directory-pathname directory))
         (json (com.inuoe.jzon:parse (merge-pathnames "tokenizer.json" dir)))
         (config (let ((f (merge-pathnames "tokenizer_config.json" dir)))
                   (and (probe-file f) (com.inuoe.jzon:parse f))))
         (model (json-get json "model"))
         (decoders (decoder-types (json-get json "decoder")))
         (byte-level (some (lambda (d) (equal (json-get d "type") "ByteLevel")) decoders)))
    (unless (equal (json-get model "type") "BPE")
      (error "Only BPE tokenizers are supported, not ~S." (json-get model "type")))
    (unless (or byte-level
                (some (lambda (d) (member (json-get d "type") '("ByteFallback" "Metaspace" "Replace")
                                          :test #'equal))
                      decoders))
      (error "Unsupported tokenizer decoder ~S." (json-get json "decoder")))
    (let* ((vocab (make-hash-table :test 'equal))
           (ranks (make-hash-table :test 'equal))
           (specials '())
           (special-ids (make-hash-table)))
      (maphash (lambda (k v) (setf (gethash k vocab) v)) (json-get model "vocab"))
      (loop for merge across (json-get model "merges")
            for rank from 0
            ;; "a b" strings, or [a, b] pairs in newer files
            do (setf (gethash (if (stringp merge)
                                  (let ((space (position #\Space merge)))
                                    (merge-key (subseq merge 0 space) (subseq merge (1+ space))))
                                  (merge-key (aref merge 0) (aref merge 1)))
                              ranks)
                     rank))
      (loop for added across (json-get json "added_tokens")
            for content = (json-get added "content")
            for id = (json-get added "id")
            do (when (json-get added "normalized")
                 ;; matched after normalization in the tokenizers library
                 (unless (string= content (funcall (parse-normalizer (json-get json "normalizer")) content))
                   (error "Unsupported added token ~S: normalized matching." content)))
               (setf (gethash content vocab) id)
               (push (make-added content id :lstrip (json-get added "lstrip") :rstrip (json-get added "rstrip"))
                     specials)
               (when (json-get added "special") (setf (gethash id special-ids) t)))
      (let ((id->token (make-array (1+ (loop for v being the hash-values of vocab maximize v))
                                   :initial-element "")))
        (maphash (lambda (k v) (setf (aref id->token v) k)) vocab)
        (flet ((config-token (key)
                 (let ((v (json-get config key)))
                   (when (hash-table-p v) (setf v (json-get v "content")))
                   (and (stringp v) v))))
          (make-instance 'tokenizer
                         :vocab vocab :id->token id->token :ranks ranks
                         :special specials
                         :special-ids special-ids
                         :pre-tokenizers (parse-pre-tokenizer (json-get json "pre_tokenizer"))
                         :normalizer (parse-normalizer (json-get json "normalizer"))
                         :byte-level byte-level
                         :byte-fallback (json-get model "byte_fallback")
                         :unk-id (let ((u (json-get model "unk_token"))) (and u (gethash u vocab)))
                         :ignore-merges (json-get model "ignore_merges")
                         :strip-leading-space (some (lambda (d) (and (equal (json-get d "type") "Strip")
                                                                     (eql (json-get d "start") 1)))
                                                    decoders)
                         :chat-template (json-get config "chat_template")
                         :bos-string (config-token "bos_token")
                         :eos-string (config-token "eos_token")
                         :bos-token (and (json-get config "add_bos_token")
                                         (gethash (config-token "bos_token") vocab))
                         :eos-tokens (let ((e (config-token "eos_token")))
                                       (and e (gethash e vocab) (list (gethash e vocab))))))))))
