;;;; llm/package.lisp

(defpackage :mlx.llm
  (:use :cl)
  (:local-nicknames (:mx :mlx) (:nn :mlx.nn) (:random :mlx.random) (:fast :mlx.fast))
  (:documentation "Language models on MLX: Hugging Face byte-level BPE tokenizers,
Llama-family transformers with a KV cache, sampling and generation.")
  (:export
   ;; tokenizer
   #:tokenizer #:load-tokenizer #:encode #:decode #:vocab-size #:token-id
   #:make-stream-decoder #:decode-step #:eos-tokens #:bos-token
   #:apply-chat-template
   ;; models
   #:load-model #:model-config #:model-tokenizer #:download-model #:resolve-model
   #:llama #:make-cache #:cache-offset
   ;; generation
   #:generate #:generate-tokens #:make-sampler))
