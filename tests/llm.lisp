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
  "Run the suite.  $MLX_CL_TEST_DEVICE=cpu runs it on the CPU."
  (let ((device (uiop:getenv "MLX_CL_TEST_DEVICE")))
    (when (and device (plusp (length device)))
      (mx:set-default-device (intern (string-upcase device) :keyword))))
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

;;; SentencePiece-style tokenizers (Gemma, Phi-3, Llama 2), built by hand

(defun spm-tokenizer (&key strip prepend (specials '()))
  "A tiny SentencePiece BPE tokenizer: characters, a few merges, byte
fallback for everything else."
  (let* ((tokens (append '("<unk>" "▁" "h" "e" "l" "o" "w" "r" "d" "he" "ll" "hell" "hello" "▁hello"
                           "▁w" "or" "▁wor" "▁world")
                         (loop for b below 256 collect (format nil "<0x~2,'0X>" b))
                         (mapcar #'first specials)))
         (vocab (make-hash-table :test 'equal))
         (ranks (make-hash-table :test 'equal)))
    (loop for tok in tokens for i from 0 do (setf (gethash tok vocab) i))
    (loop for (a b) in '(("h" "e") ("l" "l") ("he" "ll") ("hell" "o") ("▁" "hello") ("▁" "w")
                         ("o" "r") ("▁w" "or") ("▁wor" "l") ("▁worl" "d"))
          for rank from 0
          do (setf (gethash (mlx.llm::merge-key a b) ranks) rank))
    (setf (gethash (mlx.llm::merge-key "▁wor" "l") ranks) 8
          (gethash "▁worl" vocab) (length tokens))
    (let ((id->token (make-array (1+ (hash-table-count vocab)))))
      (maphash (lambda (k v) (setf (aref id->token v) k)) vocab)
      (make-instance 'llm:tokenizer
                     :vocab vocab :id->token id->token :ranks ranks
                     :special (loop for (content . flags) in specials
                                    collect (mlx.llm::make-added content (gethash content vocab)
                                                                 :lstrip (getf flags :lstrip)
                                                                 :rstrip (getf flags :rstrip)))
                     :special-ids (let ((h (make-hash-table)))
                                    (dolist (sp specials h) (setf (gethash (gethash (first sp) vocab) h) t)))
                     :byte-level nil :byte-fallback t :unk-id 0
                     :normalizer (lambda (text)
                                   (let ((text (substitute (code-char #x2581) #\Space text)))
                                     (if (and prepend (plusp (length text)))
                                         (concatenate 'string "▁" text)
                                         text)))
                     :strip-leading-space strip))))

(test sentencepiece-encoding
  (let ((tk (spm-tokenizer)))
    (flet ((toks (text) (mapcar (lambda (id) (aref (slot-value tk 'mlx.llm::id->token) id))
                                (llm:encode tk text))))
      (is (equal '("hello" "▁world") (toks "hello world")))
      ;; characters outside the vocabulary fall back to their UTF-8 bytes
      (is (equal '("hello" "<0xC3>" "<0xA9>") (toks "helloé")))
      (is (string= "hello world" (llm:decode tk (llm:encode tk "hello world"))))
      (is (string= "helloé" (llm:decode tk (llm:encode tk "helloé")))))
    ;; Phi-3/Llama 2 style: prepend ▁, strip one leading space when decoding
    (let ((phi (spm-tokenizer :prepend t :strip t)))
      (is (equal (llm:encode phi "hello") (list (llm:token-id phi "▁hello"))))
      (is (string= "hello world" (llm:decode phi (llm:encode phi "hello world"))))
      (let ((decoder (llm:make-stream-decoder phi)))
        (is (equal '("hello" " world")
                   (mapcar (lambda (id) (llm:decode-step decoder id)) (llm:encode phi "hello world"))))))))

(test added-tokens-strip-whitespace
  (let ((tk (spm-tokenizer :specials '(("<|a|>" :rstrip t) ("<|b|>" :lstrip t) ("<|c|>")))))
    (flet ((parts (text) (mlx.llm::split-special tk text)))
      (is (equal (list (llm:token-id tk "<|a|>") "hello") (parts (format nil "<|a|>  ~%hello"))))
      (is (equal (list "hello" (llm:token-id tk "<|b|>")) (parts "hello   <|b|>")))
      (is (equal (list "hello " (llm:token-id tk "<|c|>") " world") (parts "hello <|c|> world"))))))

(test split-and-metaspace-pre-tokenizers
  (is (equal '("a▁" "b▁" "c") (mlx.llm::split-on-string "a▁b▁c" "▁" :merged-with-previous)))
  (is (equal '("a" "▁b" "▁c") (mlx.llm::split-on-string "a▁b▁c" "▁" :merged-with-next)))
  (is (equal '("a" "▁" "b") (mlx.llm::split-on-string "a▁b" "▁" :isolated)))
  (is (equal '("a" "b") (mlx.llm::split-on-string "a▁b" "▁" :removed)))
  (let ((ms (code-char #x2581)))
    (is (equal (list (format nil "~Chello" ms) (format nil "~Cworld" ms))
               (mlx.llm::metaspace "hello world" ms :always t t)))
    (is (equal (list (format nil "hello~Cworld" ms))
               (mlx.llm::metaspace "hello world" ms :first nil nil)))))

(test bpe-merge-by-rank
  ;; lowest rank first, all occurrences, adjacent pairs recomputed
  (let ((ranks (make-hash-table :test 'equal)))
    (loop for (a b) in '(("a" "b") ("ab" "ab") ("c" "d")) for r from 0
          do (setf (gethash (mlx.llm::merge-key a b) ranks) r))
    (is (equal '("abab" "cd" "a") (mlx.llm::bpe-merge ranks '("a" "b" "a" "b" "c" "d" "a"))))
    (is (equal '("x") (mlx.llm::bpe-merge ranks '("x"))))
    (is (= 2000 (length (mlx.llm::bpe-merge ranks (make-list 2000 :initial-element "z")))))))

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
  (let* ((model (mlx.llm::make-causal-lm (tiny-config)))
         (tokens (loop repeat 20 collect (random 97))))
    (multiple-value-bind (diff offset) (incremental-matches-full-p model tokens 7)
      (is (< diff 1e-4) "incremental decoding differs from a full pass by ~A" diff)
      (is (= 20 offset)))))

(test kv-cache-grows-past-a-chunk
  ;; the cache allocates 256 positions at a time; cross the boundary
  (random:seed 1)
  (let* ((model (mlx.llm::make-causal-lm (tiny-config)))
         (tokens (loop repeat 270 collect (random 97))))
    (multiple-value-bind (diff offset cache) (incremental-matches-full-p model tokens 250)
      (is (< diff 1e-4))
      (is (= 270 offset))
      (is (= 512 (mx:dim (mlx.llm::kv-cache-keys (first cache)) 2)) "grown by one chunk"))))

(test model-variants-build
  (let ((qwen (mlx.llm::make-causal-lm (tiny-config "model_type" "qwen2"))))
    (is (nn:child (nn:child (nn:child (first (nn:child (nn:child qwen :model) :layers)) :self-attn) :q-proj) :bias))
    (is (null (nn:child qwen :lm-head)) "tied embeddings"))
  (let ((untied (mlx.llm::make-causal-lm (tiny-config "tie_word_embeddings" nil))))
    (is (typep (nn:child untied :lm-head) 'nn:linear))
    (is (equal '(1 3 97) (mx:shape (funcall untied (mx:from-lisp '((1 2 3)) :dtype :int32)
                                            (llm:make-cache untied)))))))

(test family-architectures-cache-consistently
  ;; for each family, decoding through the KV cache matches a full pass;
  ;; Gemma 3's sliding window is made small enough to take effect
  (loop for (name . overrides)
          in '(("phi3" "model_type" "phi3")
               ("gemma2" "model_type" "gemma2" "head_dim" 16 "query_pre_attn_scalar" 16
                "attn_logit_softcapping" 50.0 "final_logit_softcapping" 30.0)
               ("gemma3" "model_type" "gemma3_text" "head_dim" 16 "query_pre_attn_scalar" 16
                "sliding_window" 6 "sliding_window_pattern" 2 "rope_local_base_freq" 10000.0))
        do (random:seed 3)
           (let* ((model (mlx.llm::make-causal-lm (apply #'tiny-config overrides)))
                  (tokens (loop repeat 20 collect (random 97))))
             (is (< (incremental-matches-full-p model tokens 7) 1e-4) "~A cache consistency" name)))
  (let ((arch (mlx.llm::model-arch (mlx.llm::make-causal-lm (tiny-config "model_type" "gemma3_text")))))
    (is (mlx.llm::arch-qk-norm arch))
    (is (mlx.llm::arch-four-norms arch))
    (is (eq :bf16 (mlx.llm::arch-embed-scale arch))))
  (signals error (mlx.llm::make-causal-lm (tiny-config "model_type" "mamba"))))

(test switch-layers
  (random:seed 5)
  (let* ((sl (nn:switch-linear 8 6 4 :bias t))
         (x (random:normal :shape '(3 1 8)))
         (idx (mx:from-lisp '((0 2) (3 3) (1 0)) :dtype :uint32))
         (y (funcall sl (mx:expand-dims x -2) idx)))
    (is (equal '(3 2 1 6) (mx:shape y)))
    ;; each (token, slot) equals the chosen expert's plain linear map
    (loop for tk below 3
          do (loop for slot below 2
                   for e = (mx:item (mx:ref idx tk slot))
                   do (is (close-enough (mx:ref y tk slot 0)
                                        (mx:add (mx:matmul (mx:ref x tk 0) (mx:transpose (mx:ref (nn:child sl "weight") e)))
                                                (mx:ref (nn:child sl "bias") e)))
                           "token ~D slot ~D" tk slot))))
  ;; SwitchGLU: sorted dispatch (>= 64 indices) and unsorted agree
  (let* ((glu (nn:switch-glu 16 8 8))
         (x (random:normal :shape '(20 16)))
         (idx (mx:astype (random:randint 0 8 :shape '(20 4)) :uint32))
         (sorted (funcall glu x idx))
         (unsorted (mx:concatenate (loop for i below 20 collect (funcall glu (mx:ref x (list i (1+ i))) (mx:ref idx (list i (1+ i))))))))
    (is (equal '(20 4 16) (mx:shape sorted)))
    (is (close-enough sorted unsorted)))
  ;; quantizing converts switch layers too
  (let ((m (nn:sequential (nn:switch-linear 64 32 4))))
    (nn:quantize m)
    (is (typep (first (nn:child m "layers")) 'nn:quantized-switch-linear))))

(defun close-enough (a b &optional (tol 1e-3))
  (< (mx:item (mx:max (mx:abs (mx:subtract (mx:astype a :float32) (mx:astype b :float32))))) tol))

(test moe-configuration
  (let* ((config (tiny-config "model_type" "qwen3_moe" "num_experts" 4 "num_experts_per_tok" 2
                              "moe_intermediate_size" 16 "decoder_sparse_step" 1
                              "mlp_only_layers" #(1) "head_dim" 16))
         (model (mlx.llm::make-causal-lm config))
         (layers (nn:child (nn:child model :model) :layers)))
    (is (typep (nn:child (first layers) "mlp") 'mlx.llm::moe-block))
    (is (typep (nn:child (second layers) "mlp") 'mlx.llm::mlp) "mlp_only_layers stay dense"))
  (let ((mixtral (mlx.llm::make-causal-lm (tiny-config "model_type" "mixtral" "num_local_experts" 4
                                                        "num_experts_per_tok" 2))))
    (is (nn:child (first (nn:child (nn:child mixtral :model) :layers)) "block_sparse_moe"))))

(test hugging-face-expert-weights-are-stacked
  (let* ((alist (loop for e below 3
                      nconc (loop for (w . n) in '(("w1" . 1) ("w2" . 2) ("w3" . 3))
                                  collect (cons (format nil "model.layers.0.block_sparse_moe.experts.~D.~A.weight" e w)
                                                (mx:full '(2 2) (+ (* 10 e) n))))))
         (stacked (mlx.llm::sanitize-weights (append alist (list (cons "model.norm.weight" (mx:ones '(2))))))))
    (is (= 4 (length stacked)))
    (let ((gate (cdr (assoc "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight" stacked :test #'string=)))
          (down (cdr (assoc "model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight" stacked :test #'string=))))
      (is (equal '(3 2 2) (mx:shape gate)))
      (is (equal '(1 11 21) (mapcar (lambda (e) (round (mx:item (mx:ref gate e 0 0)))) '(0 1 2))) "w1 -> gate_proj, in expert order")
      (is (= 2 (round (mx:item (mx:ref down 0 0 0)))) "w2 -> down_proj")
      ;; the per-expert arrays are released once stacked: holding both
      ;; doubled the memory needed to load OLMoE (7.5 GB instead of 3.9)
      (is (every (lambda (e) (mx:freed-p (cdr e))) alist)))))

;;; Tiny random models made by mlx-lm (tools/make-model-fixtures.py): the
;;; same weights must give mlx-lm's logits

(defparameter *tiny-tokens* '(5 17 3 42 8 60 1 33 21 9 50 12 7 44 2 30))

(test tiny-models-match-mlx-lm
  (let* ((root (asdf:system-relative-pathname "mlx" "tests/fixtures/tiny/"))
         (fixture-version (string-trim '(#\Newline) (uiop:read-file-string (merge-pathnames "MLX_VERSION" root))))
         (gpu (eq :gpu (mx:device-type (mx:default-device))))
         ;; bit-exact only with the kernels that made the fixtures: same MLX
         ;; version, same Apple GPU generation ($MLX_CL_TEST_EXACT asserts it)
         (exact (and gpu (exact-generation-p) (string= fixture-version (mx:version)))))
    (dolist (name '("qwen3" "mixtral" "qwen2_moe" "qwen3_moe" "olmoe" "mixtral-4bit"))
      (let* ((dir (merge-pathnames (format nil "~A/" name) root))
             (model (llm:load-model dir :tokenizer nil))
             (unquantized-moe (and (not (search "4bit" name)) (not (string= name "qwen3")))))
        (if (and (not gpu) unquantized-moe)
            ;; MLX's CPU gather_mm supports only float32 weights
            (skip "~A: bf16 experts need the GPU" name)
            (let* ((full (mx:ref (funcall model (mx:from-lisp (list *tiny-tokens*) :dtype :int32)
                                          (llm:make-cache model))
                                 0))
                   (cache (llm:make-cache model))
                   (rows (list (mx:ref (funcall model (mx:from-lisp (list (subseq *tiny-tokens* 0 12)) :dtype :int32)
                                                cache)
                                       0 -1))))
              (dolist (id (subseq *tiny-tokens* 12 15))
                (push (mx:ref (funcall model (mx:from-lisp (list (list id)) :dtype :int32) cache) 0 -1) rows))
              (loop for (what ours file) in `(("full pass" ,full "logits-full.npy")
                                              ("incremental" ,(mx:stack (nreverse rows)) "logits-incremental.npy"))
                    for diff = (mx:item (mx:max (mx:abs (mx:subtract (mx:astype ours :float32)
                                                                     (mx:load (merge-pathnames file dir))))))
                    do (if exact
                           (is (zerop diff) "~A ~A differs from mlx-lm by ~A" name what diff)
                           (is (< diff 0.25) "~A ~A differs from mlx-lm by ~A" name what diff)))))))))

(test causal-mask-windows
  (is (equal '((t nil nil) (t t nil) (t t t)) (lisp (mlx.llm::causal-mask 3 0))))
  (is (equal '((t t t nil) (t t t t)) (lisp (mlx.llm::causal-mask 2 2))))
  (is (equal '((nil t t nil) (nil nil t t)) (lisp (mlx.llm::causal-mask 2 2 :window 2)))))

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
  (let* ((model (mlx.llm::make-causal-lm (tiny-config)))
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

;;; ------------------------------------------------------------------
;;; More model families against mlx-lm (opt-in: $MLX_CL_TEST_ALL_MODELS,
;;; ~9 GB of downloads: Qwen2.5, Qwen3, Llama 3.2, Gemma 2, Gemma 3,
;;; Phi-3.5, OLMoE)

(test model-families-match-mlx-lm
  (if (not (let ((v (uiop:getenv "MLX_CL_TEST_ALL_MODELS"))) (and v (plusp (length v)))))
      (skip "set MLX_CL_TEST_ALL_MODELS to test Qwen2.5, Qwen3, Llama 3.2, Gemma 2/3, Phi-3.5 and OLMoE")
      ;; one model in memory at a time: together they exceed 16 GB machines
      (let ((current-repo nil) (current nil))
        (dolist (case (fixture "families.sexp"))
          (destructuring-bind (&key repo messages (thinking t) text prompt-ids eos ids) case
            (let* ((model (if (equal repo current-repo)
                              current
                              (progn (when current (mx:free (nn:parameters current)))
                                     (setf current-repo repo current (llm:load-model repo)))))
                   (tk (llm:model-tokenizer model))
                   ;; Llama 3 templates print today's date
                   (text (let ((p (search "Today Date: " text)))
                           (if p
                               (concatenate 'string (subseq text 0 (+ p 12)) (mlx.llm::today-string)
                                            (subseq text (position #\Newline text :start p)))
                               text)))
                   (want (remove-if (lambda (id) (member id eos)) ids))
                   (got '()))
              (is (string= text (llm:apply-chat-template tk messages :thinking thinking)) "~A chat template" repo)
              (unless (search "Today Date" text)
                (is (equal prompt-ids (llm:encode tk text)) "~A prompt tokens" repo))
              (llm:generate-tokens model prompt-ids (lambda (id) (push id got))
                                   :max-tokens (length ids) :eos-ids eos)
              (setf got (nreverse got))
              (if (exact-generation-p)
                  (is (equal want got) "~A greedy tokens (first difference at ~A)" repo (mismatch want got))
                  (is (equal (subseq want 0 10) (subseq got 0 (min 10 (length got))))
                      "~A first greedy tokens" repo))))))))
