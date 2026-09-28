;;;; llm/tokenizer.lisp -- byte-level BPE tokenizers from Hugging Face
;;;; tokenizer.json files (GPT-2, SmolLM, Llama 3, Qwen2 and relatives)
;;;;
;;;; Pre-tokenization regexes are implemented as hand-written scanners that
;;;; follow the regex semantics exactly (alternation order, greedy runs,
;;;; the (?!\S) lookahead), so no regex engine is needed.  Configurations
;;;; this file does not implement are rejected rather than approximated,
;;;; since a slightly wrong tokenizer silently degrades a model.

(in-package :mlx.llm)

(defclass tokenizer ()
  ((vocab :initarg :vocab :documentation "token string -> id")
   (id->token :initarg :id->token :documentation "vector id -> token string")
   (ranks :initarg :ranks :documentation "\"left right\" -> merge rank")
   (special :initarg :special :documentation "list of (content . id), longest first")
   (special-ids :initarg :special-ids :documentation "hash id -> t for special tokens")
   (pre-tokenizers :initarg :pre-tokenizers :documentation "list of :digits / (:regex style)")
   (normalizer :initarg :normalizer :initform #'identity)
   (cache :initform (make-hash-table :test 'equal))
   (chat-template :initarg :chat-template :initform nil :reader tokenizer-chat-template)
   (bos-token :initarg :bos-token :initform nil :reader bos-token)
   (eos-tokens :initarg :eos-tokens :initform '() :accessor eos-tokens
               :documentation "ids that end generation")))

(defmethod print-object ((tk tokenizer) stream)
  (print-unreadable-object (tk stream :type t)
    (format stream "~:D tokens" (hash-table-count (slot-value tk 'vocab)))))

(defun vocab-size (tokenizer) (length (slot-value tokenizer 'id->token)))

(defun token-id (tokenizer token)
  "The id of the token string TOKEN, or NIL."
  (values (gethash token (slot-value tokenizer 'vocab))))

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

(defun pre-tokenize (tokenizer text)
  (let ((pieces (list text)))
    (dolist (step (slot-value tokenizer 'pre-tokenizers) pieces)
      (setf pieces
            (mapcan (lambda (piece)
                      (ecase (if (consp step) (first step) step)
                        (:digits (split-digits piece))
                        (:gpt2 (scan piece #'gpt2-match-end))
                        (:llama3 (let ((max (second step)))
                                   (scan piece (lambda (s i) (llama3-match-end s i max)))))))
                    pieces)))))

;;; ------------------------------------------------------------------
;;; BPE

(defun bpe (tokenizer word)
  "Token strings for the byte-level-mapped WORD."
  (let ((cache (slot-value tokenizer 'cache)))
    (or (gethash word cache)
        (setf (gethash word cache)
              (let ((ranks (slot-value tokenizer 'ranks))
                    (symbols (map 'list #'string word)))
                (loop
                  (let ((best nil) (best-rank nil))
                    (loop for (a b) on symbols
                          while b
                          do (let ((r (gethash (concatenate 'string a " " b) ranks)))
                               (when (and r (or (null best-rank) (< r best-rank)))
                                 (setf best-rank r best (cons a b)))))
                    (unless best (return symbols))
                    ;; merge every occurrence of the best pair
                    (setf symbols
                          (loop with out = '()
                                while symbols
                                do (let ((a (pop symbols)))
                                     (if (and symbols (string= a (car best)) (string= (first symbols) (cdr best)))
                                         (push (concatenate 'string a (pop symbols)) out)
                                         (push a out)))
                                finally (return (nreverse out)))))))))))

(defun encode-plain (tokenizer text)
  (let ((vocab (slot-value tokenizer 'vocab)))
    (loop for piece in (pre-tokenize tokenizer (funcall (slot-value tokenizer 'normalizer) text))
          nconc (loop for token in (bpe tokenizer (bytes->bpe-string
                                                   (sb-ext:string-to-octets piece :external-format :utf-8)))
                      collect (or (gethash token vocab)
                                  (error "Token ~S is not in the vocabulary." token))))))

(defun split-special (tokenizer text)
  "Split TEXT into strings and special-token ids."
  (let ((specials (slot-value tokenizer 'special))
        (out '()) (start 0) (i 0) (n (length text)))
    (loop while (< i n)
          do (let ((hit (find-if (lambda (sp) (let ((end (+ i (length (car sp)))))
                                                (and (<= end n) (string= (car sp) text :start2 i :end2 end))))
                                 specials)))
               (if hit
                   (progn (when (> i start) (push (subseq text start i) out))
                          (push (cdr hit) out)
                          (incf i (length (car hit)))
                          (setf start i))
                   (incf i))))
    (when (< start n) (push (subseq text start) out))
    (nreverse out)))

(defun encode (tokenizer text &key add-bos)
  "Token ids for TEXT.  Special tokens written in TEXT (e.g. <|im_start|>)
become their ids.  ADD-BOS prepends the beginning-of-sequence token."
  (append (and add-bos (bos-token tokenizer) (list (bos-token tokenizer)))
          (loop for part in (split-special tokenizer text)
                nconc (if (integerp part) (list part) (encode-plain tokenizer part)))))

;;; ------------------------------------------------------------------
;;; Decoding

(defun token-octets (tokenizer id)
  (let ((token (aref (slot-value tokenizer 'id->token) id)))
    (if (gethash id (slot-value tokenizer 'special-ids))
        (sb-ext:string-to-octets token :external-format :utf-8)
        (map '(vector (unsigned-byte 8)) (lambda (c) (gethash c *char->byte* 63)) token))))

(defun decode (tokenizer ids &key skip-special)
  "The text of the token IDS."
  (let ((octets (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
    (dolist (id ids)
      (unless (and skip-special (gethash id (slot-value tokenizer 'special-ids)))
        (loop for b across (token-octets tokenizer id) do (vector-push-extend b octets))))
    (sb-ext:octets-to-string (coerce octets '(vector (unsigned-byte 8)))
                             :external-format '(:utf-8 :replacement #\?))))

(defstruct (stream-decoder (:constructor make-stream-decoder (tokenizer)))
  tokenizer
  (pending (make-array 0 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))

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
  (let ((pending (stream-decoder-pending decoder)))
    (loop for b across (token-octets (stream-decoder-tokenizer decoder) id)
          do (vector-push-extend b pending))
    (let* ((complete (utf8-complete-length pending))
           (text (sb-ext:octets-to-string (coerce (subseq pending 0 complete) '(vector (unsigned-byte 8)))
                                          :external-format '(:utf-8 :replacement #\?)))
           (rest (subseq pending complete)))
      (setf (fill-pointer pending) 0)
      (loop for b across rest do (vector-push-extend b pending))
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

(defun parse-normalizer (spec)
  "A function normalizing text for the tokenizer.json normalizer SPEC."
  (if (null spec)
      #'identity
      (let ((type (json-get spec "type")))
        (cond ((string= type "NFC") (lambda (s) (sb-unicode:normalize-string s :nfc)))
              ((string= type "NFKC") (lambda (s) (sb-unicode:normalize-string s :nfkc)))
              ((string= type "Sequence")
               (let ((steps (map 'list #'parse-normalizer (json-get spec "normalizers"))))
                 (lambda (s) (reduce (lambda (acc f) (funcall f acc)) steps :initial-value s))))
              (t (error "Unsupported normalizer type ~S." type))))))

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
        ((string= type "Split")
         (let* ((pattern (json-get spec "pattern" "Regex"))
                (known (assoc pattern *llama3-patterns* :test #'equal)))
           (cond ((equal pattern *gpt2-pattern*) (list :gpt2))
                 (known (list (list :llama3 (cdr known))))
                 (t (error "Unsupported Split pre-tokenizer pattern: ~S" pattern)))))
        (t (error "Unsupported pre-tokenizer type ~S." type))))))

(defun load-tokenizer (directory)
  "Load the tokenizer.json (and tokenizer_config.json) in DIRECTORY."
  (let* ((dir (uiop:ensure-directory-pathname directory))
         (json (com.inuoe.jzon:parse (merge-pathnames "tokenizer.json" dir)))
         (config (let ((f (merge-pathnames "tokenizer_config.json" dir)))
                   (and (probe-file f) (com.inuoe.jzon:parse f))))
         (model (json-get json "model")))
    (unless (equal (json-get model "type") "BPE")
      (error "Only BPE tokenizers are supported, not ~S." (json-get model "type")))
    (let* ((vocab (make-hash-table :test 'equal))
           (ranks (make-hash-table :test 'equal))
           (specials '())
           (special-ids (make-hash-table)))
      (maphash (lambda (k v) (setf (gethash k vocab) v)) (json-get model "vocab"))
      (loop for merge across (json-get model "merges")
            for rank from 0
            ;; "a b" strings, or [a, b] pairs in newer files
            do (setf (gethash (if (stringp merge) merge (format nil "~A ~A" (aref merge 0) (aref merge 1)))
                              ranks)
                     rank))
      (loop for added across (json-get json "added_tokens")
            for content = (json-get added "content")
            for id = (json-get added "id")
            do (setf (gethash content vocab) id)
               (push (cons content id) specials)
               (when (json-get added "special") (setf (gethash id special-ids) t)))
      (let ((id->token (make-array (1+ (loop for v being the hash-values of vocab maximize v))
                                   :initial-element "")))
        (maphash (lambda (k v) (setf (aref id->token v) k)) vocab)
        (flet ((config-token (key)
                 (let ((v (json-get config key)))
                   (when (hash-table-p v) (setf v (json-get v "content")))
                   (and (stringp v) (gethash v vocab)))))
          (make-instance 'tokenizer
                         :vocab vocab :id->token id->token :ranks ranks
                         :special (sort specials #'> :key (lambda (s) (length (car s))))
                         :special-ids special-ids
                         :pre-tokenizers (parse-pre-tokenizer (json-get json "pre_tokenizer"))
                         :normalizer (parse-normalizer (json-get json "normalizer"))
                         :chat-template (json-get config "chat_template")
                         :bos-token (and (json-get config "add_bos_token") (config-token "bos_token"))
                         :eos-tokens (remove nil (list (config-token "eos_token")))))))))
