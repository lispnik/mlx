;;;; llm/models.lisp -- decoder-only transformers of the Llama lineage
;;;;
;;;; One configurable implementation covers Hugging Face model types llama,
;;;; mistral, qwen2, qwen3, phi3, gemma2, gemma3_text, and the mixture-of-
;;;; experts types mixtral, qwen2_moe, qwen3_moe and olmoe.  An ARCH struct, derived
;;;; from config.json, records how a family departs from Llama: norm style,
;;;; fused projections, MLP activation, embedding scaling, soft-capping,
;;;; sliding windows...  Each variation follows mlx-lm's implementation of
;;;; that family operation for operation, so outputs match it exactly.
;;;; Child names match the checkpoints ("model.layers.0.self_attn.q_proj"),
;;;; so weights load directly with NN:LOAD-WEIGHTS.

(in-package :mlx.llm)

(defun config-get (config key &optional default)
  (let ((v (json-get config key)))
    (if (null v) default v)))

;;; ------------------------------------------------------------------
;;; Architecture description

(defstruct arch
  type
  (norm-offset nil)       ; Gemma RMSNorm scales by (1 + weight)
  (embed-scale nil)       ; multiply embeddings by sqrt(hidden): t, or :bf16 to
                          ; round the factor to bfloat16 first (Gemma 3)
  (clip-residual nil)     ; float16 residual adds in float32, clipped (Gemma 3)
  (qk-norm nil)           ; RMSNorm of queries and keys: :gemma3 (per head, after
                          ; the transpose), :per-head (before it, Qwen3) or
                          ; :full (over all heads, before the reshape, OLMoE)
  (moe nil)               ; mixture-of-experts plist, see CONFIG-MOE
  (four-norms nil)        ; pre/post norms around both attention and MLP (Gemma 2/3)
  (fused-qkv nil)         ; one qkv_proj (Phi-3)
  (fused-gate-up nil)     ; one gate_up_proj (Phi-3)
  (activation :silu)      ; MLP gate: :silu (SwiGLU) or :gelu-approx (GeGLU)
  (attn-softcap nil)      ; tanh soft-capping of attention scores (Gemma 2)
  (final-softcap nil)     ; ... and of the output logits
  (sliding-window nil)    ; window size of local-attention layers (Gemma 3)
  (sliding-pattern nil)   ; every Nth layer is global
  (qkv-bias nil) (o-bias nil) (mlp-bias nil))

(defparameter *supported-model-types*
  '("llama" "mistral" "qwen2" "qwen3" "phi3" "gemma2" "gemma3_text"
    "mixtral" "qwen2_moe" "qwen3_moe" "olmoe"))

(defun config-moe (config type)
  "The mixture-of-experts description for TYPE, or NIL.  :ROUTING says how
experts are chosen, as each family does in mlx-lm:
  :mixtral        top-k of the router logits, then a softmax over those k
  :softmax-first  softmax over all experts, then top-k (argpartition of -p)
  :qwen3          softmax, then top-k from the other end (argpartition of p)"
  (flet ((cfg (key &optional default) (config-get config key default)))
    (cond
      ((equal type "mixtral")
       (list :name "block_sparse_moe" :routing :mixtral :experts (cfg "num_local_experts")
             :top-k (cfg "num_experts_per_tok") :hidden (cfg "intermediate_size")))
      ((equal type "olmoe")
       (list :name "mlp" :routing :softmax-first :experts (cfg "num_experts")
             :top-k (cfg "num_experts_per_tok") :hidden (cfg "intermediate_size")
             :norm-topk (cfg "norm_topk_prob") :flatten t :bias (cfg "mlp_bias")))
      ((equal type "qwen2_moe")
       (list :name "mlp" :routing :softmax-first :experts (cfg "num_experts")
             :top-k (cfg "num_experts_per_tok") :hidden (cfg "moe_intermediate_size")
             :shared (cfg "shared_expert_intermediate_size")))
      ((equal type "qwen3_moe")
       (list :name "mlp" :routing :qwen3 :experts (cfg "num_experts")
             :top-k (cfg "num_experts_per_tok") :hidden (cfg "moe_intermediate_size")
             :norm-topk (cfg "norm_topk_prob") :sparse-step (cfg "decoder_sparse_step" 1)
             :mlp-only (coerce (cfg "mlp_only_layers" #()) 'list))))))

(defun moe-layer-p (arch index)
  "Whether layer INDEX is a mixture-of-experts layer (else a dense MLP)."
  (let ((moe (arch-moe arch)))
    (and moe
         (not (member index (getf moe :mlp-only)))
         (plusp (getf moe :experts))
         (zerop (mod (1+ index) (getf moe :sparse-step 1))))))

(defun config-arch (config)
  (let ((type (config-get config "model_type")))
    (unless (member type *supported-model-types* :test #'equal)
      (error "Unsupported model_type ~S (supported: ~{~A~^, ~})." type *supported-model-types*))
    (let ((gemma (member type '("gemma2" "gemma3_text") :test #'equal)))
      (make-arch :type type
                 :norm-offset gemma
                 :embed-scale (and gemma (if (equal type "gemma3_text") :bf16 t))
                 :clip-residual (equal type "gemma3_text")
                 :qk-norm (cond ((equal type "gemma3_text") :gemma3)
                                ((member type '("qwen3" "qwen3_moe") :test #'equal) :per-head)
                                ((equal type "olmoe") :full))
                 :moe (config-moe config type)
                 :four-norms gemma
                 :fused-qkv (equal type "phi3")
                 :fused-gate-up (equal type "phi3")
                 :activation (if gemma :gelu-approx :silu)
                 :attn-softcap (and (equal type "gemma2") (config-get config "attn_logit_softcapping" 50.0))
                 :final-softcap (and (equal type "gemma2") (config-get config "final_logit_softcapping" 30.0))
                 :sliding-window (and (equal type "gemma3_text") (config-get config "sliding_window" 512))
                 :sliding-pattern (and (equal type "gemma3_text") (config-get config "sliding_window_pattern" 6))
                 :qkv-bias (or (member type '("qwen2" "qwen2_moe") :test #'equal)
                               (config-get config "attention_bias" nil))
                 :o-bias (and (not (member type '("qwen2" "qwen2_moe") :test #'equal))
                              (config-get config "attention_bias" nil))
                 :mlp-bias (config-get config "mlp_bias" nil)))))

(defun head-dim (config)
  (or (config-get config "head_dim")
      (/ (config-get config "hidden_size") (config-get config "num_attention_heads"))))

(defun sliding-layer-p (arch index)
  (and (arch-sliding-pattern arch)
       (/= (mod (1+ index) (arch-sliding-pattern arch)) 0)))

;;; ------------------------------------------------------------------
;;; KV cache: keys/values pre-allocated in chunks, filled with slice
;;; updates, so each step costs O(new tokens) rather than a concatenation.

(defconstant +cache-chunk+ 256)

(defstruct (kv-cache (:constructor make-kv-cache (&optional padding)))
  (keys nil) (values nil) (offset 0)
  (padding nil))  ; for a batch: an int32 array (B 1 1 1) of each row's left padding

(defun make-cache (model &key padding)
  "A fresh KV cache (one entry per layer) for MODEL.  For a batch of
left-padded prompts, PADDING lists each row's number of padding tokens."
  (let ((padding (and padding (some #'plusp padding)
                      (mx:persist (mx:reshape (mx:from-lisp padding :dtype :int32) (list -1 1 1 1))))))
    (loop repeat (length (nn:child (nn:child model :model) :layers)) collect (make-kv-cache padding))))

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

(defun padded-mask (n offset padding &key window)
  "Boolean (B 1 N OFFSET+N) mask for left-padded rows: the causal mask,
with each row's padding keys hidden.  A padding query still sees itself,
so no row is fully masked (which would make NaNs that leak into values)."
  (let* ((keys (mx:reshape (mx:arange (+ offset n)) (list 1 1 1 -1)))
         (queries (mx:reshape (mx:arange offset (+ offset n)) (list 1 1 -1 1))))
    (mx:logical-and (causal-mask n offset :window window)
                    (mx:logical-or (mx:greater-equal keys padding) (mx:equal keys queries)))))

(defun causal-mask (n offset &key window)
  "Boolean (N, OFFSET+N) mask: query i sees key j when j <= OFFSET+i (and,
with WINDOW, OFFSET+i < j + WINDOW) -- mlx-lm's create_causal_mask."
  (let* ((rinds (mx:expand-dims (mx:arange (+ offset n)) 0))
         (linds (mx:expand-dims (mx:arange offset (+ offset n)) 1))
         (mask (mx:greater-equal linds rinds)))
    (if window
        (mx:logical-and mask (mx:less linds (mx:add rinds window)))
        mask)))

;;; ------------------------------------------------------------------
;;; Norms

(nn:defmodule rms-norm* ()
  ((eps :initarg :eps) (offset :initarg :offset)))

(defun make-norm (dims eps arch)
  "RMSNorm; Gemma's variant scales by (1 + weight)."
  (let ((m (make-instance 'rms-norm* :eps eps :offset (arch-norm-offset arch))))
    (nn:register m "weight" (mx:ones (list dims)))
    m))

(defmethod nn:forward ((m rms-norm*) &rest args)
  (destructuring-bind (x) args
    (with-slots (eps offset) m
      (let ((w (nn:child m "weight")))
        (fast:rms-norm x :weight (if offset (mx:add 1.0 w) w) :eps eps)))))

;;; ------------------------------------------------------------------
;;; Rotary embeddings: plain, linear-scaled, Llama 3, and Phi-3's LongRoPE

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
  ((dims :initarg :dims) (base :initarg :base) (scale :initarg :scale :initform 1.0)
   (traditional :initarg :traditional :initform nil) (freqs :initarg :freqs :initform nil)
   (input-scale :initarg :input-scale :initform nil)))

(defun make-rotary (config dims &key base)
  "The rotary embedding for CONFIG over DIMS features; BASE overrides rope_theta."
  (let ((base (float (or base (config-get config "rope_theta" 10000.0)) 1.0))
        (scaling (config-get config "rope_scaling"))
        (traditional (config-get config "rope_traditional" nil)))
    (let ((type (and scaling (or (config-get scaling "type") (config-get scaling "rope_type")))))
      (cond ((or (null type) (equal type "default"))
             (make-instance 'rotary :dims dims :base base :traditional traditional))
            ((string= type "linear")
             (make-instance 'rotary :dims dims :base base :traditional traditional
                                    :scale (/ 1.0 (config-get scaling "factor"))))
            ((string= type "llama3")
             (make-instance 'rotary :dims dims :base nil :traditional traditional
                                    :freqs (mx:persist (llama3-rope-frequencies dims base scaling))))
            ((member type '("longrope" "su") :test #'equal)
             ;; as mlx-lm's SuScaledRoPE: long_factor frequencies, and the
             ;; input scaled by sqrt(1 + ln(factor) / ln(original context))
             (let* ((original (config-get config "original_max_position_embeddings" 4096))
                    (factor (/ (config-get config "max_position_embeddings" 131072) original))
                    (freqs (mx:multiply (mx:from-lisp (coerce (config-get scaling "long_factor") 'list)
                                                      :dtype :float32)
                                        (mx:power base (mx:divide (mx:arange 0 dims 2 :dtype :float32)
                                                                  dims)))))
               (make-instance 'rotary :dims dims :base nil :freqs (mx:persist freqs)
                                      :input-scale (if (<= factor 1)
                                                       nil
                                                       (sqrt (+ 1 (/ (log (float factor 1d0))
                                                                     (log (float original 1d0)))))))))
            (t (error "Unsupported rope_scaling type ~S." type))))))

(defmethod nn:forward ((m rotary) &rest args)
  (destructuring-bind (x offset) args
    (with-slots (dims base scale traditional freqs input-scale) m
      (let ((x (cond ((null input-scale) x)
                     ((= dims (mx:dim x -1)) (mx:multiply x input-scale))
                     (t (let ((y (mx:copy x)))
                          (setf (mx:ref y t t t (list 0 dims)) (mx:multiply (mx:ref y t t t (list 0 dims)) input-scale))
                          y)))))
        (fast:rope x dims :traditional traditional :base base :scale scale :offset offset
                          :freqs freqs)))))

;;; ------------------------------------------------------------------
;;; Attention

(nn:defmodule attention ()
  ((arch :initarg :arch) (heads :initarg :heads) (kv-heads :initarg :kv-heads)
   (head-dim :initarg :head-dim) (scale :initarg :scale) (rotary :initarg :rotary)
   (window :initarg :window :initform nil)))

(defun make-attention (config arch index)
  (let* ((dim (config-get config "hidden_size"))
         (heads (config-get config "num_attention_heads"))
         (kv-heads (config-get config "num_key_value_heads" heads))
         (head-dim (head-dim config))
         (sliding (sliding-layer-p arch index))
         (scale (float (expt (float (if (arch-norm-offset arch)
                                        (config-get config "query_pre_attn_scalar" head-dim)
                                        head-dim)
                                    1d0)
                             -0.5d0)
                       1.0))
         (m (make-instance 'attention
                           :arch arch :heads heads :kv-heads kv-heads :head-dim head-dim :scale scale
                           :window (and sliding (arch-sliding-window arch))
                           :rotary (make-rotary config
                                                (floor (* head-dim (config-get config "partial_rotary_factor" 1.0)))
                                                :base (and sliding (config-get config "rope_local_base_freq"))))))
    (if (arch-fused-qkv arch)
        (nn:register m :qkv-proj (nn:linear dim (* (+ heads (* 2 kv-heads)) head-dim) :bias nil))
        (progn
          (nn:register m :q-proj (nn:linear dim (* heads head-dim) :bias (arch-qkv-bias arch)))
          (nn:register m :k-proj (nn:linear dim (* kv-heads head-dim) :bias (arch-qkv-bias arch)))
          (nn:register m :v-proj (nn:linear dim (* kv-heads head-dim) :bias (arch-qkv-bias arch)))))
    (nn:register m :o-proj (nn:linear (* heads head-dim) dim :bias (arch-o-bias arch)))
    (when (arch-qk-norm arch)
      (let ((eps (config-get config "rms_norm_eps" 1e-6))
            (full (eq (arch-qk-norm arch) :full)))
        (nn:register m :q-norm (make-norm (if full (* heads head-dim) head-dim) eps arch))
        (nn:register m :k-norm (make-norm (if full (* kv-heads head-dim) head-dim) eps arch))))
    m))

(defun dtype-min (dtype)
  "The most negative finite value of the float DTYPE (numpy's finfo.min)."
  (ecase dtype
    (:float32 most-negative-single-float)
    (:float16 -65504.0)
    (:bfloat16 (sb-kernel:make-single-float (- #xFF7F0000 (expt 2 32))))))

(defun softcapped-attention (q k v scale cap mask heads kv-heads)
  "Attention with tanh soft-capped scores, computed explicitly (the fused
kernel has no soft-capping) exactly as mlx-lm's Gemma 2 does."
  (destructuring-bind (b h len d) (mx:shape q)
    (declare (ignore h))
    (let* ((repeats (/ heads kv-heads))
           (q (mx:multiply q scale))
           (q (if (> repeats 1) (mx:reshape q (list b kv-heads repeats len d)) q))
           (k (if (> repeats 1) (mx:expand-dims k 2) k))
           (v (if (> repeats 1) (mx:expand-dims v 2) v))
           (scores (mx:matmul q (mx:swapaxes k -1 -2)))
           (scores (mx:multiply (mx:tanh (mx:divide scores cap)) cap))
           (scores (if mask (mx:where mask scores (dtype-min (mx:dtype scores))) scores))
           (scores (mx:softmax scores :precise t :axis -1))
           (out (mx:matmul scores v)))
      (if (> repeats 1) (mx:reshape out (list b heads len d)) out))))

(defmethod nn:forward ((m attention) &rest args)
  (destructuring-bind (x cache) args
    (with-slots (arch heads kv-heads head-dim scale rotary window) m
      (destructuring-bind (b len dim) (mx:shape x)
        (declare (ignore dim))
        (flet ((heads-first (y n &optional norm)
                 ;; (B L n*D) -> (B n L D), normalizing where the family does
                 (let* ((qk (arch-qk-norm arch))
                        (y (if (and norm (eq qk :full)) (funcall (nn:child m norm) y) y))
                        (y (mx:reshape y (list b len n head-dim)))
                        (y (if (and norm (eq qk :per-head)) (funcall (nn:child m norm) y) y))
                        (y (mx:transpose y :axes '(0 2 1 3))))
                   (if (and norm (eq qk :gemma3)) (funcall (nn:child m norm) y) y))))
          (multiple-value-bind (q k v)
              (if (arch-fused-qkv arch)
                  (destructuring-bind (q k v)
                      (mx:split (funcall (nn:child m :qkv-proj) x)
                                (list (* heads head-dim) (* (+ heads kv-heads) head-dim)) :axis -1)
                    (values q k v))
                  (values (funcall (nn:child m :q-proj) x) (funcall (nn:child m :k-proj) x)
                          (funcall (nn:child m :v-proj) x)))
            (let* ((q (heads-first q heads :q-norm))
                   (k (heads-first k kv-heads :k-norm))
                   (v (heads-first v kv-heads))
                   (offset (kv-cache-offset cache))
                   (q (funcall rotary q offset))
                   (k (funcall rotary k offset)))
              (multiple-value-bind (keys values) (cache-update cache k v)
                (let* ((padding (kv-cache-padding cache))
                       (out
                        (cond
                          (padding
                           (let ((mask (padded-mask len offset padding :window window)))
                             (if (arch-attn-softcap arch)
                                 (softcapped-attention q keys values scale (arch-attn-softcap arch)
                                                       mask heads kv-heads)
                                 (fast:scaled-dot-product-attention q keys values scale
                                                                    :mask-mode "array" :mask-arr mask))))
                          ((arch-attn-softcap arch)
                           (softcapped-attention q keys values scale (arch-attn-softcap arch)
                                                 (and (> len 1) (causal-mask len offset))
                                                 heads kv-heads))
                          ;; a sliding window reaches back past the start
                          ((and window (> (+ offset len) window))
                           (fast:scaled-dot-product-attention q keys values scale
                                                              :mask-mode "array"
                                                              :mask-arr (causal-mask len offset :window window)))
                          (t (fast:scaled-dot-product-attention q keys values scale
                                                                :mask-mode (if (> len 1) "causal" ""))))))
                  (funcall (nn:child m :o-proj)
                           (mx:reshape (mx:transpose out :axes '(0 2 1 3))
                                       (list b len (* heads head-dim)))))))))))))

;;; ------------------------------------------------------------------
;;; MLP and blocks

(nn:defmodule mlp () ((arch :initarg :arch)))

(defun make-mlp (config arch &optional hidden)
  (let ((dim (config-get config "hidden_size"))
        (hidden (or hidden (config-get config "intermediate_size")))
        (bias (arch-mlp-bias arch))
        (m (make-instance 'mlp :arch arch)))
    (if (arch-fused-gate-up arch)
        (nn:register m :gate-up-proj (nn:linear dim (* 2 hidden) :bias bias))
        (progn (nn:register m :gate-proj (nn:linear dim hidden :bias bias))
               (nn:register m :up-proj (nn:linear dim hidden :bias bias))))
    (nn:register m :down-proj (nn:linear hidden dim :bias bias))
    m))

(defmethod nn:forward ((m mlp) &rest args)
  (destructuring-bind (x) args
    (let ((arch (slot-value m 'arch)))
      (multiple-value-bind (gate up)
          (if (arch-fused-gate-up arch)
              (destructuring-bind (g u) (mx:split (funcall (nn:child m :gate-up-proj) x) 2 :axis -1)
                (values g u))
              (values (funcall (nn:child m :gate-proj) x) (funcall (nn:child m :up-proj) x)))
        (funcall (nn:child m :down-proj)
                 (ecase (arch-activation arch)
                   (:silu (nn:swiglu gate up))
                   (:gelu-approx (mx:multiply (nn:gelu-approx gate) up))))))))

;;; Mixture of experts: a router picks TOP-K experts per token; their outputs
;;; are combined with the routing weights (mlx-lm's *SparseMoeBlock classes)

(nn:defmodule moe-block () ((moe :initarg :moe)))

(defun make-moe-block (config arch)
  (let* ((moe (arch-moe arch))
         (dim (config-get config "hidden_size"))
         (m (make-instance 'moe-block :moe moe)))
    (nn:register m "gate" (nn:linear dim (getf moe :experts) :bias nil))
    (nn:register m "switch_mlp" (nn:switch-glu dim (getf moe :hidden) (getf moe :experts)
                                               :bias (getf moe :bias)))
    (when (getf moe :shared)
      (nn:register m "shared_expert" (make-mlp config arch (getf moe :shared)))
      (nn:register m "shared_expert_gate" (nn:linear dim 1 :bias nil)))
    m))

(defun first-k (a k)
  "A[..., :k]"
  (apply #'mx:ref a (append (make-list (1- (mx:ndim a)) :initial-element t) (list (list 0 k)))))

(defun last-k (a k)
  "A[..., -k:]"
  (apply #'mx:ref a (append (make-list (1- (mx:ndim a)) :initial-element t) (list (list (- k) nil)))))

(defun route (moe gates)
  "Returns (values expert-indices scores) for router logits GATES."
  (let ((k (getf moe :top-k)))
    (flet ((normalized (scores)
             (if (getf moe :norm-topk)
                 (mx:divide scores (mx:sum scores :axis -1 :keepdims t))
                 scores)))
      (ecase (getf moe :routing)
        (:mixtral
         (let* ((inds (mx:stop-gradient (first-k (mx:argpartition (mx:negative gates) (1- k) :axis -1) k)))
                (scores (mx:take-along-axis gates inds -1)))
           (values inds (mx:softmax scores :axis -1 :precise t))))
        (:softmax-first
         (let* ((probs (mx:softmax gates :axis -1 :precise t))
                (inds (mx:stop-gradient (first-k (mx:argpartition (mx:negative probs) (1- k) :axis -1) k))))
           (values inds (normalized (mx:take-along-axis probs inds -1)))))
        (:qwen3
         (let* ((probs (mx:softmax gates :axis -1 :precise t))
                (inds (last-k (mx:argpartition probs (- k) :axis -1) k)))
           (values inds (normalized (mx:take-along-axis probs inds -1)))))))))

(defmethod nn:forward ((m moe-block) &rest args)
  (destructuring-bind (x) args
    (let* ((moe (slot-value m 'moe))
           (shape (mx:shape x))
           (x (if (getf moe :flatten) (mx:reshape x (list -1 (car (last shape)))) x)))
      (multiple-value-bind (inds scores) (route moe (funcall (nn:child m "gate") x))
        (let* ((y (funcall (nn:child m "switch_mlp") x inds))
               (y (mx:sum (mx:multiply y (mx:expand-dims scores -1)) :axis -2))
               (y (if (nn:child m "shared_expert")
                      (mx:add y (mx:multiply (mx:sigmoid (funcall (nn:child m "shared_expert_gate") x))
                                             (funcall (nn:child m "shared_expert") x)))
                      y)))
          (if (getf moe :flatten) (mx:reshape y shape) y))))))

(nn:defmodule transformer-block () ((arch :initarg :arch) (ffn :initarg :ffn :initform "mlp")))

(defun make-block (config arch index)
  (let* ((moe (moe-layer-p arch index))
         (ffn (if moe (getf (arch-moe arch) :name) "mlp"))
         (m (make-instance 'transformer-block :arch arch :ffn ffn))
         (dim (config-get config "hidden_size"))
         (eps (config-get config "rms_norm_eps" 1e-5)))
    (nn:register m :self-attn (make-attention config arch index))
    (nn:register m ffn (if moe (make-moe-block config arch) (make-mlp config arch)))
    (nn:register m :input-layernorm (make-norm dim eps arch))
    (nn:register m :post-attention-layernorm (make-norm dim eps arch))
    (when (arch-four-norms arch)
      (nn:register m :pre-feedforward-layernorm (make-norm dim eps arch))
      (nn:register m :post-feedforward-layernorm (make-norm dim eps arch)))
    m))

(defun clip-residual (x y)
  "x + y; for float16, computed in float32 and clipped to the float16 range
(Gemma 3, as mlx-lm)."
  (if (eq (mx:dtype x) :float16)
      (mx:astype (mx:clip (mx:add (mx:astype x :float32) (mx:astype y :float32))
                          :a-min -65504.0 :a-max 65504.0)
                 :float16)
      (mx:add x y)))

(defmethod nn:forward ((m transformer-block) &rest args)
  (destructuring-bind (x cache) args
    (flet ((c (name) (nn:child m name)))
      (let ((attn (funcall (c :self-attn) (funcall (c :input-layernorm) x) cache))
            (add (if (arch-clip-residual (slot-value m 'arch)) #'clip-residual #'mx:add)))
        (if (arch-four-norms (slot-value m 'arch))
            (let ((h (funcall add x (funcall (c :post-attention-layernorm) attn))))
              (funcall add h (funcall (c :post-feedforward-layernorm)
                                      (funcall (c (slot-value m 'ffn)) (funcall (c :pre-feedforward-layernorm) h)))))
            (let ((h (mx:add x attn)))
              (mx:add h (funcall (c (slot-value m 'ffn)) (funcall (c :post-attention-layernorm) h)))))))))

;;; ------------------------------------------------------------------
;;; The model

(nn:defmodule decoder () ())

(nn:defmodule causal-lm ()
  ((config :initarg :config :reader model-config)
   (arch :initarg :arch :reader model-arch)
   (tokenizer :initarg :tokenizer :initform nil :accessor model-tokenizer)))

(defun make-causal-lm (config &key (tied (config-get config "tie_word_embeddings" nil)))
  "An untrained model for the parsed config.json CONFIG."
  (let* ((arch (config-arch config))
         (m (make-instance 'causal-lm :config config :arch arch))
         (decoder (make-instance 'decoder))
         (dim (config-get config "hidden_size")))
    (nn:register decoder :embed-tokens (nn:embedding (config-get config "vocab_size") dim))
    (nn:register decoder :layers (loop for i below (config-get config "num_hidden_layers")
                                       collect (make-block config arch i)))
    (nn:register decoder :norm (make-norm dim (config-get config "rms_norm_eps" 1e-5) arch))
    (nn:register m :model decoder)
    (unless tied
      (nn:register m :lm-head (nn:linear dim (config-get config "vocab_size") :bias nil)))
    m))

(defmethod nn:forward ((m causal-lm) &rest args)
  "(funcall model tokens cache): TOKENS is a (B L) int array; returns
logits (B L vocab) and advances CACHE."
  (destructuring-bind (tokens cache) args
    (let* ((arch (model-arch m))
           (decoder (nn:child m :model))
           (h (funcall (nn:child decoder :embed-tokens) tokens))
           (scale (sqrt (float (config-get (model-config m) "hidden_size") 1d0)))
           (h (case (arch-embed-scale arch)
                ((nil) h)
                ;; as mlx-lm: mx.array(sqrt(hidden), bfloat16).astype(h.dtype)
                (:bf16 (mx:multiply h (mx:astype (mx:scalar scale :dtype :bfloat16) (mx:dtype h))))
                (t (mx:multiply h scale)))))
      (loop for layer in (nn:child decoder :layers)
            for c in cache
            do (setf h (funcall layer h c)))
      (setf h (funcall (nn:child decoder :norm) h))
      (let ((logits (if (nn:child m :lm-head)
                        (funcall (nn:child m :lm-head) h)
                        (nn:as-linear (nn:child decoder :embed-tokens) h)))
            (cap (arch-final-softcap arch)))
        (if cap
            (mx:multiply (mx:tanh (mx:divide logits cap)) cap)
            logits)))))

(defmethod mlx.nn.impl::module-description ((m causal-lm))
  (let ((c (model-config m)))
    (format nil "~A, ~D layers, ~D dims"
            (config-get c "model_type") (config-get c "num_hidden_layers") (config-get c "hidden_size"))))

;;; ------------------------------------------------------------------
;;; Loading

(defparameter *expert-projections*
  '(("w1" . "gate_proj") ("w2" . "down_proj") ("w3" . "up_proj")   ; Mixtral
    ("gate_proj" . "gate_proj") ("up_proj" . "up_proj") ("down_proj" . "down_proj")))

(defun expert-key (name)
  "For a per-expert weight \"<prefix>.experts.<n>.<proj>.<suffix>\" (Hugging
Face layout), return (values stacked-name n); else NIL."
  (let ((p (search ".experts." name)))
    (when p
      (let* ((rest (subseq name (+ p 9)))
             (dot (position #\. rest))
             (n (and dot (parse-integer rest :end dot :junk-allowed t)))
             (tail (and n (subseq rest (1+ dot))))
             (proj-end (and tail (position #\. tail)))
             (proj (and proj-end (cdr (assoc (subseq tail 0 proj-end) *expert-projections* :test #'string=)))))
        (when proj
          (values (format nil "~A.switch_mlp.~A~A" (subseq name 0 p) proj (subseq tail proj-end)) n))))))

(defun sanitize-weights (alist)
  "Drop checkpoint entries with no counterpart in the model, and stack
Hugging Face per-expert weights (experts.N.*) into the switch_mlp layout,
as mlx-lm's sanitize does.  Some checkpoints, even mlx-community ones such
as OLMoE's, keep the per-expert layout."
  (let ((groups (make-hash-table :test 'equal)) (out '()))
    (dolist (e alist)
      (multiple-value-bind (stacked n) (expert-key (car e))
        (cond ((search "rotary_emb.inv_freq" (car e)))
              (stacked (push (cons n (cdr e)) (gethash stacked groups)))
              (t (push e out)))))
    (maphash (lambda (name experts)
               (let ((parts (mapcar #'cdr (sort experts #'< :key #'car))))
                 (push (cons name (mx:stack parts)) out)
                 ;; the stack's graph holds what it needs; dropping our own
                 ;; handles now lets MLX free each expert's copy once it is
                 ;; stacked, instead of keeping both until a GC
                 (mx:free parts)))
             groups)
    (nreverse out)))

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
         (weights (load-weight-alist dir))
         ;; a checkpoint without lm_head ties the output to the embedding
         (model (make-causal-lm config :tied (not (assoc "lm_head.weight" weights :test #'string=)))))
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
      (add-config-eos (model-tokenizer model) config dir)
      (add-turn-terminators (model-tokenizer model)))
    model))

(defparameter *turn-terminators* '("<end_of_turn>" "<|end|>" "<|eot_id|>" "<|im_end|>")
  "Tokens that close a chat turn in the templates APPLY-CHAT-TEMPLATE renders.")

(defun add-turn-terminators (tokenizer)
  "Also stop at the chat template's end-of-turn token.  Some checkpoints
(e.g. Gemma and Phi-3 conversions) leave it out of eos_token_id, and
generation then runs on past the end of the reply."
  (let ((template (tokenizer-chat-template tokenizer)))
    (dolist (token *turn-terminators*)
      (let ((id (token-id tokenizer token)))
        (when (and id template (search token template))
          (pushnew id (eos-tokens tokenizer)))))))

(defun add-config-eos (tokenizer config dir)
  "End generation on every eos_token_id in config.json and generation_config.json."
  (let ((gen (let ((f (merge-pathnames "generation_config.json" dir)))
               (and (probe-file f) (com.inuoe.jzon:parse f)))))
    (dolist (source (list config gen))
      (let ((ids (config-get source "eos_token_id")))
        (dolist (id (if (vectorp ids) (coerce ids 'list) (list ids)))
          (when (integerp id) (pushnew id (eos-tokens tokenizer))))))))
