;;;; llm/package.lisp

(defpackage :mlx.llm
  (:use :cl)
  (:local-nicknames (:mx :mlx) (:nn :mlx.nn) (:random :mlx.random) (:fast :mlx.fast))
  (:documentation "Language models on MLX: Hugging Face BPE tokenizers (byte-level
and SentencePiece), Llama, Qwen2, Mistral, Phi-3 and Gemma 2/3 transformers
with a KV cache, sampling and generation.")
  (:export
   ;; tokenizer
   #:tokenizer #:load-tokenizer #:encode #:decode #:vocab-size #:token-id
   #:make-stream-decoder #:decode-step #:eos-tokens #:bos-token
   #:apply-chat-template
   ;; models
   #:load-model #:model-config #:model-tokenizer #:download-model #:resolve-model
   #:causal-lm #:model-arch #:make-cache #:cache-offset
   ;; generation
   #:generate #:generate-tokens #:make-sampler
   ;; writing Lisp
   #:generate-lisp-form #:evaluate-lisp #:write-lisp))
