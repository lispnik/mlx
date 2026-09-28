;;;; tests/llm.lisp -- mlx/llm: tokenizers, the Llama model, generation
;;;;
;;;; Most tests need no downloads: pre-tokenizer scanners are checked
;;;; against splits produced by the Hugging Face tokenizers library
;;;; (fixtures/pretokenize.sexp), and a tiny random model checks the KV
;;;; cache.  Tests against real weights (SmolLM2-135M-Instruct, ~270 MB)
;;;; run only when $MLX_CL_TEST_MODELS is set; they compare against
;;;; tokenizers and mlx-lm output (fixtures/smollm.sexp).  Full-length greedy
;;;; agreement is checked only with $MLX_CL_TEST_EXACT, since bf16 results
;;;; differ between Apple GPU generations.

(defpackage :mlx-llm-tests
  (:use :cl :fiveam)
  (:local-nicknames (:mx :mlx) (:nn :mlx.nn) (:random :mlx.random) (:llm :mlx.llm))
  (:export #:run-tests))

(in-package :mlx-llm-tests)

(def-suite :mlx.llm :description "mlx/llm tests.")
(in-suite :mlx.llm)

(defun run-tests ()
  (let ((results (run :mlx.llm)))
    (explain! results)
    (results-status results)))

(defun fixture (name)
  (with-open-file (in (asdf:system-relative-pathname "mlx" (format nil "tests/fixtures/~A" name))
                      :external-format :utf-8)
    (let ((*read-eval* nil)) (read in))))

(defun lisp (a) (mx:to-lisp a :as :list))

;;; ------------------------------------------------------------------
;;; Tokenizer pieces

(defun tokenizer-with (&rest initargs)
  (apply #'make-instance 'llm:tokenizer
         (append initargs (list :vocab (make-hash-table :test 'equal) :id->token #()
                                :ranks (make-hash-table :test 'equal) :special '()
                                :special-ids (make-hash-table)))))

(test pre-tokenizers-match-hugging-face
  (loop for (name . cases) in (fixture "pretokenize.sexp")
        for steps = (cond ((string= name "smol") '(:digits :gpt2))
                          ((string= name "qwen") '((:llama3 1)))
                          (t '((:llama3 3))))
        for tokenizer = (tokenizer-with :pre-tokenizers steps)
        do (loop for (text . expected) in cases
                 for ours = (mapcar (lambda (piece)
                                      (mlx.llm::bytes->bpe-string
                                       (sb-ext:string-to-octets piece :external-format :utf-8)))
                                    (mlx.llm::pre-tokenize tokenizer text))
                 do (is (equal expected ours) "~A pre-tokenization of ~S" name text))))

(test byte-level-mapping-roundtrips
  (let ((all (make-array 256 :element-type '(unsigned-byte 8) :initial-contents (loop for i below 256 collect i))))
    (is (equalp all (map '(vector (unsigned-byte 8)) (lambda (c) (gethash c mlx.llm::*char->byte*))
                         (mlx.llm::bytes->bpe-string all))))
    (is (string= "Ġ" (mlx.llm::bytes->bpe-string #(32))))))

(defun byte-tokenizer ()
  "A tokenizer whose token i is the single byte i."
  (let ((id->token (make-array 256)))
    (dotimes (i 256) (setf (aref id->token i) (string (aref mlx.llm::*byte->char* i))))
    (make-instance 'llm:tokenizer :vocab (make-hash-table :test 'equal) :id->token id->token
                                  :ranks (make-hash-table :test 'equal) :special '()
                                  :special-ids (make-hash-table) :pre-tokenizers '())))

(test streaming-decode-holds-partial-utf8
  (let* ((tk (byte-tokenizer))
         (text "é👍x")
         (bytes (coerce (sb-ext:string-to-octets text :external-format :utf-8) 'list))
         (decoder (llm:make-stream-decoder tk))
         (pieces (mapcar (lambda (b) (llm:decode-step decoder b)) bytes)))
    ;; é is 2 bytes, 👍 is 4: text appears only when a character completes
    (is (equal '("" "é" "" "" "" "👍" "x") pieces))
    (is (string= text (llm:decode tk bytes)))))

(test chat-templates
  (let ((chatml (tokenizer-with :chat-template "{{ '<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
' }}{{'<|im_start|>' + message['role'] }}"))
        (escaped (tokenizer-with :chat-template "{{- '<|im_start|>system\\n' + messages[0]['content'] }}{%- else %}{{- '<|im_start|>system\\nYou are Qwen.<|im_end|>\\n' }}"))
        (llama3 (tokenizer-with :chat-template "<|start_header_id|> Cutting Knowledge Date"))
        (unknown (tokenizer-with :chat-template "{{ messages }}")))
    (is (string= (format nil "<|im_start|>system~%You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>~%<|im_start|>user~%Hi<|im_end|>~%<|im_start|>assistant~%")
                 (llm:apply-chat-template chatml '(("user" . "Hi")))))
    (is (string= (format nil "<|im_start|>system~%Be brief.<|im_end|>~%<|im_start|>user~%Hi<|im_end|>~%")
                 (llm:apply-chat-template chatml '(("system" . "Be brief.") ("user" . "Hi"))
                                          :add-generation-prompt nil)))
    (is (search (format nil "system~%You are Qwen.<|im_end|>") (llm:apply-chat-template escaped '(("user" . "Hi")))))
    (let ((text (llm:apply-chat-template llama3 '(("user" . "Hi")))))
      (is (search "<|begin_of_text|><|start_header_id|>system<|end_header_id|>" text))
      (is (search (format nil "Today Date: ~A" (mlx.llm::today-string)) text))
      (is (search (format nil "<|start_header_id|>user<|end_header_id|>~%~%Hi<|eot_id|>") text)))
    (signals error (llm:apply-chat-template unknown '(("user" . "Hi"))))))

;;; ------------------------------------------------------------------
;;; Model mechanics on a tiny random Llama (no download)

(defun tiny-config (&rest overrides)
  (let ((h (make-hash-table :test 'equal)))
    (loop for (k v) on (append overrides
                               '("model_type" "llama" "hidden_size" 64 "num_attention_heads" 4
                                 "num_key_value_heads" 2 "intermediate_size" 128 "num_hidden_layers" 2
                                 "vocab_size" 97 "rms_norm_eps" 1e-5 "rope_theta" 10000.0
                                 "tie_word_embeddings" t))
          by #'cddr
          do (unless (nth-value 1 (gethash k h)) (setf (gethash k h) v)))
    h))

(defun incremental-matches-full-p (model tokens prefill)
  "Logits for TOKENS from one full pass versus a PREFILL-token prefill
followed by single-token steps through the cache."
  (let* ((full (mx:ref (funcall model (mx:from-lisp (list tokens) :dtype :int32) (llm:make-cache model)) 0))
         (cache (llm:make-cache model))
         (first-rows (mx:ref (funcall model (mx:from-lisp (list (subseq tokens 0 prefill)) :dtype :int32) cache) 0))
         (rest (loop for id in (nthcdr prefill tokens)
                     collect (mx:ref (funcall model (mx:from-lisp (list (list id)) :dtype :int32) cache) 0 0)))
         (incremental (mx:concatenate (cons first-rows (list (mx:stack rest))) :axis 0)))
    (values (mx:item (mx:max (mx:abs (mx:subtract full incremental))))
            (llm:cache-offset cache)
            cache)))

(test kv-cache-matches-full-forward
  (random:seed 0)
  (let* ((model (mlx.llm::make-llama (tiny-config)))
         (tokens (loop repeat 20 collect (random 97))))
    (multiple-value-bind (diff offset) (incremental-matches-full-p model tokens 7)
      (is (< diff 1e-4) "incremental decoding differs from a full pass by ~A" diff)
      (is (= 20 offset)))))

(test kv-cache-grows-past-a-chunk
  ;; the cache allocates 256 positions at a time; cross the boundary
  (random:seed 1)
  (let* ((model (mlx.llm::make-llama (tiny-config)))
         (tokens (loop repeat 270 collect (random 97))))
    (multiple-value-bind (diff offset cache) (incremental-matches-full-p model tokens 250)
      (is (< diff 1e-4))
      (is (= 270 offset))
      (is (= 512 (mx:dim (mlx.llm::kv-cache-keys (first cache)) 2)) "grown by one chunk"))))

(test model-variants-build
  (let ((qwen (mlx.llm::make-llama (tiny-config "model_type" "qwen2"))))
    (is (nn:child (nn:child (nn:child (first (nn:child (nn:child qwen :model) :layers)) :self-attn) :q-proj) :bias))
    (is (null (nn:child qwen :lm-head)) "tied embeddings"))
  (let ((untied (mlx.llm::make-llama (tiny-config "tie_word_embeddings" nil))))
    (is (typep (nn:child untied :lm-head) 'nn:linear))
    (is (equal '(1 3 97) (mx:shape (funcall untied (mx:from-lisp '((1 2 3)) :dtype :int32)
                                            (llm:make-cache untied)))))))

(test llama3-rope-frequencies-match-mlx-lm
  (let* ((scaling (let ((h (make-hash-table :test 'equal)))
                    (setf (gethash "factor" h) 32.0 (gethash "high_freq_factor" h) 4.0
                          (gethash "low_freq_factor" h) 1.0
                          (gethash "original_max_position_embeddings" h) 8192)
                    h))
         (freqs (lisp (mlx.llm::llama3-rope-frequencies 64 500000.0 scaling))))
    (is (= 32 (length freqs)))
    ;; reference values from mlx-lm's Llama3RoPE
    (loop for (i expected) in '((0 1.0) (1 1.5069) (2 2.2708) (14 311.39) (15 774.86)
                                (16 2327.98) (17 10300.48) (29 4675645.5) (31 10617621.0))
          do (is (< (abs (- (nth i freqs) expected)) (* 1e-3 expected)) "frequency ~D" i))))

(test sampling
  (let ((logits (mx:from-lisp '((0.0 5.0 1.0) (9.0 0.0 0.0)))))
    (is (equal '(1 0) (lisp (funcall (llm:make-sampler) logits))))
    ;; a tiny nucleus keeps only the most likely token
    (is (equal '(1 0) (lisp (funcall (llm:make-sampler :temperature 1.0 :top-p 0.01) logits))))
    (random:seed 0)
    (let ((draws (loop repeat 50 collect (first (lisp (funcall (llm:make-sampler :temperature 1.0)
                                                              (mx:from-lisp '((0.0 0.0)))))))))
      (is (and (member 0 draws) (member 1 draws)) "temperature sampling varies"))))

(test generation-stops-and-streams
  (random:seed 0)
  (let* ((model (mlx.llm::make-llama (tiny-config)))
         (seen '()))
    (is (= 5 (llm:generate-tokens model '(1 2 3) (lambda (id) (push id seen)) :max-tokens 5)))
    (is (= 5 (length seen)))
    (let ((first-id (car (last seen))) (count 0))
      ;; greedy is deterministic, so the same first token is produced again;
      ;; declaring it end-of-sequence stops immediately
      (is (zerop (llm:generate-tokens model '(1 2 3) (lambda (id) (declare (ignore id)) (incf count))
                                      :max-tokens 5 :eos-ids (list first-id))))
      (is (zerop count)))))

;;; ------------------------------------------------------------------
;;; Against real weights (opt-in: set $MLX_CL_TEST_MODELS)

(defparameter *model-repo* "HuggingFaceTB/SmolLM2-135M-Instruct")

(defvar *model* nil)

(defun model-tests-enabled-p ()
  (let ((v (uiop:getenv "MLX_CL_TEST_MODELS")))
    (and v (plusp (length v)))))

(defun exact-generation-p ()
  "True when full greedy runs should match the fixtures exactly: set
$MLX_CL_TEST_EXACT on hardware like the one they were produced on (M3)."
  (let ((v (uiop:getenv "MLX_CL_TEST_EXACT")))
    (and v (plusp (length v)))))

(defmacro model-test (name &body body)
  `(test ,name
     (if (model-tests-enabled-p)
         (let ((*model* (or *model* (setf *model* (llm:load-model *model-repo*)))))
           ,@body)
         (skip "set MLX_CL_TEST_MODELS to test against ~A" *model-repo*))))

(model-test tokenizer-matches-hugging-face
  (let ((tk (llm:model-tokenizer *model*)))
    (loop for (text . ids) in (getf (fixture "smollm.sexp") :encodings)
          do (is (equal ids (llm:encode tk text)) "encoding of ~S" text)
             (is (string= text (llm:decode tk ids)) "decoding of ~S" text))))

(model-test greedy-generation-matches-mlx-lm
  (let ((tk (llm:model-tokenizer *model*)))
    (loop for (prompt . expected) in (getf (fixture "smollm.sexp") :greedy)
          for want = (remove-if (lambda (id) (member id (llm:eos-tokens tk))) expected)
          for got = (let ((ids '()))
                      (llm:generate-tokens *model* (llm:encode tk (llm:apply-chat-template tk (list (cons "user" prompt))))
                                           (lambda (id) (push id ids))
                                           :max-tokens 120 :eos-ids (llm:eos-tokens tk))
                      (nreverse ids))
          do (if (exact-generation-p)
                 (is (equal want got) "greedy tokens for ~S (first difference at ~A)" prompt (mismatch want got))
                 ;; bf16 kernels differ across Apple GPU generations, so long
                 ;; greedy runs eventually diverge on other hardware; the
                 ;; opening tokens are robust
                 (is (equal (subseq want 0 10) (subseq got 0 (min 10 (length got))))
                     "first greedy tokens for ~S" prompt)))))

(model-test generate-text
  (let ((text (llm:generate *model* "What is the capital of France? Answer in one sentence." :max-tokens 30)))
    (is (search "Paris" text))))
