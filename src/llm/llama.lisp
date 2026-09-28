;;;; llm/llama.lisp -- Llama-family decoder-only transformers
;;;;
;;;; Covers Hugging Face model_type llama, mistral, qwen2 and the models
;;;; built on them (SmolLM, TinyLlama...).  Child names match the
;;;; checkpoints ("model.layers.0.self_attn.q_proj.weight"), so weights load
;;;; directly with NN:LOAD-WEIGHTS.

(in-package :mlx.llm)

(defun config-get (config key &optional default)
  (let ((v (json-get config key)))
    (if (null v) default v)))

;;; ------------------------------------------------------------------
;;; KV cache: keys/values pre-allocated in chunks, filled with slice
;;; updates, so each step costs O(new tokens) rather than a concatenation.

(defconstant +cache-chunk+ 256)

(defstruct (kv-cache (:constructor make-kv-cache ()))
  (keys nil) (values nil) (offset 0))

(defun make-cache (model)
  "A fresh KV cache (one entry per layer) for MODEL."
  (loop repeat (length (nn:child (nn:child model :model) :layers)) collect (make-kv-cache)))

(defun cache-offset (cache) (kv-cache-offset (first cache)))

(defun cache-update (cache keys values)
  "Append KEYS and VALUES (B heads L dim); return all cached (keys values)."
  (let* ((prev (kv-cache-offset cache))
         (len (mx:dim keys 2))
         (end (+ prev len)))
    (when (or (null (kv-cache-keys cache)) (> end (mx:dim (kv-cache-keys cache) 2)))
      ;; grow by whole chunks
      (destructuring-bind (b h l d) (mx:shape keys)
        (declare (ignore l))
        (let* ((grow (* +cache-chunk+ (ceiling len +cache-chunk+)))
               (new-k (mx:zeros (list b h grow d) :dtype (mx:dtype keys)))
               (new-v (mx:zeros (list b h grow (mx:dim values 3)) :dtype (mx:dtype values))))
          (if (kv-cache-keys cache)
              (setf (kv-cache-keys cache)
                    (mx:concatenate (list (mx:ref (kv-cache-keys cache) t t (list 0 prev)) new-k) :axis 2)
                    (kv-cache-values cache)
                    (mx:concatenate (list (mx:ref (kv-cache-values cache) t t (list 0 prev)) new-v) :axis 2))
              (setf (kv-cache-keys cache) new-k
                    (kv-cache-values cache) new-v)))))
    (setf (mx:ref (kv-cache-keys cache) t t (list prev end)) keys
          (mx:ref (kv-cache-values cache) t t (list prev end)) values)
    (mx:persist (kv-cache-keys cache) (kv-cache-values cache))
    (setf (kv-cache-offset cache) end)
    (values (mx:ref (kv-cache-keys cache) t t (list 0 end))
            (mx:ref (kv-cache-values cache) t t (list 0 end)))))

;;; ------------------------------------------------------------------
;;; Rotary embeddings, including Llama 3 frequency scaling

(defun llama3-rope-frequencies (dims base scaling)
  (let* ((factor (config-get scaling "factor" 8.0))
         (low (config-get scaling "low_freq_factor" 1.0))
         (high (config-get scaling "high_freq_factor" 4.0))
         (old-context (config-get scaling "original_max_position_embeddings" 8192))
         (low-wavelen (/ old-context low))
         (high-wavelen (/ old-context high))
         (freqs (mx:power base (mx:divide (mx:arange 0 dims 2 :dtype :float32) dims)))
         (wavelens (mx:multiply freqs (* 2 pi)))
         (freqs (mx:where (mx:greater wavelens low-wavelen) (mx:multiply freqs factor) freqs))
         (medium (mx:logical-and (mx:greater wavelens high-wavelen) (mx:less wavelens low-wavelen)))
         (smooth (mx:divide (mx:subtract (mx:divide old-context wavelens) low) (- high low)))
         (smooth-freqs (mx:divide freqs (mx:add (mx:divide (mx:subtract 1 smooth) factor) smooth))))
    (mx:where medium smooth-freqs freqs)))

(nn:defmodule rotary ()
  ((dims :initarg :dims) (base :initarg :base) (scale :initarg :scale)
   (traditional :initarg :traditional) (freqs :initarg :freqs :initform nil)))

(defun make-rotary (config head-dim)
  (let ((base (float (config-get config "rope_theta" 10000.0) 1.0))
        (scaling (config-get config "rope_scaling"))
        (traditional (config-get config "rope_traditional" nil)))
    (let ((type (and scaling (or (config-get scaling "rope_type") (config-get scaling "type")))))
      (cond ((null type)
             (make-instance 'rotary :dims head-dim :base base :scale 1.0 :traditional traditional))
            ((string= type "linear")
             (make-instance 'rotary :dims head-dim :base base :traditional traditional
                                    :scale (/ 1.0 (config-get scaling "factor"))))
            ((string= type "llama3")
             (make-instance 'rotary :dims head-dim :base nil :scale 1.0 :traditional traditional
                                    :freqs (mx:persist (llama3-rope-frequencies head-dim base scaling))))
            (t (error "Unsupported rope_scaling type ~S." type))))))

(defmethod nn:forward ((m rotary) &rest args)
  (destructuring-bind (x offset) args
    (with-slots (dims base scale traditional freqs) m
      (fast:rope x dims :traditional traditional :base base :scale scale :offset offset
                        :freqs freqs))))

;;; ------------------------------------------------------------------
;;; Transformer blocks

(nn:defmodule attention ()
  ((heads :initarg :heads) (kv-heads :initarg :kv-heads) (head-dim :initarg :head-dim)
   (rotary :initarg :rotary)))

(defun make-attention (config)
  (let* ((dim (config-get config "hidden_size"))
         (heads (config-get config "num_attention_heads"))
         (kv-heads (config-get config "num_key_value_heads" heads))
         (head-dim (config-get config "head_dim" (/ dim heads)))
         (qwen (equal (config-get config "model_type") "qwen2"))
         (qkv-bias (or qwen (config-get config "attention_bias" nil)))
         (m (make-instance 'attention :heads heads :kv-heads kv-heads :head-dim head-dim
                                      :rotary (make-rotary config head-dim))))
    (nn:register m :q-proj (nn:linear dim (* heads head-dim) :bias qkv-bias))
    (nn:register m :k-proj (nn:linear dim (* kv-heads head-dim) :bias qkv-bias))
    (nn:register m :v-proj (nn:linear dim (* kv-heads head-dim) :bias qkv-bias))
    (nn:register m :o-proj (nn:linear (* heads head-dim) dim
                                      :bias (and (not qwen) (config-get config "attention_bias" nil))))
    m))

(defmethod nn:forward ((m attention) &rest args)
  (destructuring-bind (x cache) args
    (with-slots (heads kv-heads head-dim rotary) m
      (destructuring-bind (b len dim) (mx:shape x)
        (declare (ignore dim))
        (flet ((project (name n)
                 (mx:transpose (mx:reshape (funcall (nn:child m name) x) (list b len n head-dim))
                               :axes '(0 2 1 3))))
          (let* ((offset (kv-cache-offset cache))
                 (q (funcall rotary (project :q-proj heads) offset))
                 (k (funcall rotary (project :k-proj kv-heads) offset))
                 (v (project :v-proj kv-heads)))
            (multiple-value-bind (keys values) (cache-update cache k v)
              (let ((out (fast:scaled-dot-product-attention
                          q keys values (/ 1.0 (sqrt head-dim))
                          ;; one new token attends to everything cached
                          :mask-mode (if (> len 1) "causal" ""))))
                (funcall (nn:child m :o-proj)
                         (mx:reshape (mx:transpose out :axes '(0 2 1 3))
                                     (list b len (* heads head-dim))))))))))))

(nn:defmodule mlp () ())

(defun make-mlp (config)
  (let ((dim (config-get config "hidden_size"))
        (hidden (config-get config "intermediate_size"))
        (bias (config-get config "mlp_bias" nil))
        (m (make-instance 'mlp)))
    (nn:register m :gate-proj (nn:linear dim hidden :bias bias))
    (nn:register m :up-proj (nn:linear dim hidden :bias bias))
    (nn:register m :down-proj (nn:linear hidden dim :bias bias))
    m))

(defun swiglu (gate x)
  "silu(gate) * x as one fused kernel (as mlx-lm computes it)."
  (funcall (mlx.nn.impl::compiled 'swiglu
                                  (lambda (gate x) (mx:multiply (mx:multiply gate (mx:sigmoid gate)) x)))
           gate x))

(defmethod nn:forward ((m mlp) &rest args)
  (destructuring-bind (x) args
    (funcall (nn:child m :down-proj)
             (swiglu (funcall (nn:child m :gate-proj) x) (funcall (nn:child m :up-proj) x)))))

(nn:defmodule transformer-block () ())

(defun make-block (config)
  (let ((m (make-instance 'transformer-block))
        (dim (config-get config "hidden_size"))
        (eps (config-get config "rms_norm_eps" 1e-5)))
    (nn:register m :self-attn (make-attention config))
    (nn:register m :mlp (make-mlp config))
    (nn:register m :input-layernorm (nn:rms-norm dim :eps eps))
    (nn:register m :post-attention-layernorm (nn:rms-norm dim :eps eps))
    m))

(defmethod nn:forward ((m transformer-block) &rest args)
  (destructuring-bind (x cache) args
    (let ((h (mx:add x (funcall (nn:child m :self-attn) (funcall (nn:child m :input-layernorm) x) cache))))
      (mx:add h (funcall (nn:child m :mlp) (funcall (nn:child m :post-attention-layernorm) h))))))

;;; ------------------------------------------------------------------
;;; The model

(nn:defmodule decoder () ())

(nn:defmodule llama ()
  ((config :initarg :config :reader model-config)
   (tokenizer :initarg :tokenizer :initform nil :accessor model-tokenizer)))

(defun make-llama (config)
  "An untrained Llama-family model for the parsed config.json CONFIG."
  (let ((m (make-instance 'llama :config config))
        (decoder (make-instance 'decoder))
        (dim (config-get config "hidden_size")))
    (nn:register decoder :embed-tokens (nn:embedding (config-get config "vocab_size") dim))
    (nn:register decoder :layers (loop repeat (config-get config "num_hidden_layers")
                                       collect (make-block config)))
    (nn:register decoder :norm (nn:rms-norm dim :eps (config-get config "rms_norm_eps" 1e-5)))
    (nn:register m :model decoder)
    (unless (config-get config "tie_word_embeddings" nil)
      (nn:register m :lm-head (nn:linear dim (config-get config "vocab_size") :bias nil)))
    m))

(defmethod nn:forward ((m llama) &rest args)
  "(funcall model tokens cache): TOKENS is a (B L) int array; returns
logits (B L vocab) and advances CACHE."
  (destructuring-bind (tokens cache) args
    (let* ((decoder (nn:child m :model))
           (h (funcall (nn:child decoder :embed-tokens) tokens)))
      (loop for layer in (nn:child decoder :layers)
            for c in cache
            do (setf h (funcall layer h c)))
      (setf h (funcall (nn:child decoder :norm) h))
      (if (nn:child m :lm-head)
          (funcall (nn:child m :lm-head) h)
          (nn:as-linear (nn:child decoder :embed-tokens) h)))))

(defmethod mlx.nn.impl::module-description ((m llama))
  (let ((c (model-config m)))
    (format nil "~A, ~D layers, ~D dims"
            (config-get c "model_type") (config-get c "num_hidden_layers") (config-get c "hidden_size"))))

;;; ------------------------------------------------------------------
;;; Loading

(defparameter *supported-model-types* '("llama" "mistral" "qwen2"))

(defun sanitize-weights (alist)
  "Drop checkpoint entries with no counterpart in the model."
  (remove-if (lambda (e) (search "rotary_emb.inv_freq" (car e))) alist))

(defun load-weight-alist (dir)
  (sanitize-weights
   (loop for file in (sort (directory (merge-pathnames "*.safetensors" dir)) #'string< :key #'namestring)
         nconc (let ((table (mx:load file)))
                 (loop for k being the hash-keys of table using (hash-value v) collect (cons k v))))))

(defun load-model (source &key (tokenizer t) (lazy nil))
  "Load a model from SOURCE: a local directory holding config.json,
*.safetensors and tokenizer.json, or a Hugging Face repo id such as
\"HuggingFaceTB/SmolLM2-135M-Instruct\" (downloaded on first use; see
DOWNLOAD-MODEL).  Quantized MLX checkpoints are supported.  Unless LAZY, the
weights are evaluated before returning."
  (let* ((dir (resolve-model source))
         (config (com.inuoe.jzon:parse (merge-pathnames "config.json" dir)))
         (type (config-get config "model_type")))
    (unless (member type *supported-model-types* :test #'equal)
      (error "Unsupported model_type ~S (supported: ~{~A~^, ~})." type *supported-model-types*))
    (let ((model (make-llama config))
          (weights (load-weight-alist dir)))
      (let ((q (config-get config "quantization")))
        (when q
          ;; quantize exactly the layers the checkpoint has scales for
          (nn:quantize model :group-size (config-get q "group_size" 64) :bits (config-get q "bits" 4)
                             :mode (config-get q "mode" "affine")
                             :predicate (lambda (path m)
                                          (declare (ignore m))
                                          (assoc (format nil "~A.scales" path) weights :test #'string=)))))
      (nn:load-weights model weights)
      (unless lazy (mx:eval (nn:parameters model)))
      (when tokenizer
        (setf (model-tokenizer model) (load-tokenizer dir))
        (add-config-eos (model-tokenizer model) config dir))
      model)))

(defun add-config-eos (tokenizer config dir)
  "End generation on every eos_token_id in config.json and generation_config.json."
  (let ((gen (let ((f (merge-pathnames "generation_config.json" dir)))
               (and (probe-file f) (com.inuoe.jzon:parse f)))))
    (dolist (source (list config gen))
      (let ((ids (config-get source "eos_token_id")))
        (dolist (id (if (vectorp ids) (coerce ids 'list) (list ids)))
          (when (integerp id) (pushnew id (eos-tokens tokenizer))))))))
