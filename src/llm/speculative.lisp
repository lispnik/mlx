;;;; speculative.lisp -- speculative decoding with a small draft model
;;;;
;;;; A draft model (e.g. Qwen2.5-0.5B for Qwen2.5-7B; they share the
;;;; vocabulary) proposes K tokens one at a time; the target model then
;;;; checks all K in one forward pass, which costs about as much as one
;;;; step since decoding is bound by reading the weights.  At each position
;;;; the target samples its own token; drafts are accepted while they equal
;;;; it, and the target's token at the first mismatch comes free.  Every
;;;; emitted token is the target's own sample given the tokens before it,
;;;; so the output is what the target alone would produce -- identical when
;;;; greedy -- only faster when the draft guesses well.  Both KV caches are
;;;; rewound past rejected tokens by resetting their offsets.

(in-package :mlx.llm)

(defun rewind-cache (cache offset)
  (dolist (layer cache) (setf (kv-cache-offset layer) offset)))

(defun prefill (model ids cache &key (step 2048))
  "Run IDS through MODEL into CACHE, in chunks, discarding the logits."
  (loop for start from 0 below (length ids) by step
        do (mx:with-scope ()
             (funcall model (mx:from-lisp (list (subseq ids start (min (length ids) (+ start step)))) :dtype :int32)
                      cache)
             (mx:eval (loop for c in cache collect (kv-cache-keys c) collect (kv-cache-values c))))))

(defun generate-tokens-speculative (model draft prompt-ids function
                                    &key (max-tokens 256) (sampler (make-sampler)) eos-ids (draft-tokens 2))
  "Like GENERATE-TOKENS, with DRAFT proposing DRAFT-TOKENS tokens per
target step.  Returns (values token-count prompt-seconds
generation-seconds acceptance), ACCEPTANCE the fraction of drafts kept."
  (let* ((target-cache (make-cache model))
         (draft-cache (make-cache draft))
         (start (get-internal-real-time))
         (prompt-time nil)
         (count 0) (proposed 0) (accepted 0)
         (pending (last prompt-ids))          ; for the target: not yet in its cache
         (draft-pending (last prompt-ids)))   ; likewise for the draft
    (prefill model (butlast prompt-ids) target-cache)
    (prefill draft (butlast prompt-ids) draft-cache)
    (block generation
      (loop
        (let* ((k (max 1 (min draft-tokens (- max-tokens count 1))))
               (target-offset (cache-offset target-cache))
               (draft-offset (cache-offset draft-cache))
               ;; the draft proposes K tokens, greedily
               (drafts (let ((input draft-pending) (out '()))
                         (dotimes (i k (nreverse out))
                           (let ((d (mx:with-scope ()
                                      (mx:item (mx:argmax (mx:ref (funcall draft (mx:from-lisp (list input) :dtype :int32)
                                                                           draft-cache)
                                                                  t -1)
                                                          :axis -1)))))
                             (push d out)
                             (setf input (list d))))))
               ;; the target samples a token after each of pending, d1 .. dK
               (samples (mx:with-scope ()
                          (let* ((logits (funcall model (mx:from-lisp (list (append pending drafts)) :dtype :int32)
                                                  target-cache))
                                 (last-positions (mx:ref logits 0 (list (- (mx:dim logits 1) (1+ k)) nil))))
                            (coerce (mx:to-lisp (mx:astype (funcall sampler last-positions) :int32)) 'list))))
               (n (loop for d in drafts for s in samples while (= d s) count t)))
          (unless prompt-time
            (setf prompt-time (/ (- (get-internal-real-time) start) internal-time-units-per-second)
                  start (get-internal-real-time)))
          (incf proposed k)
          (incf accepted n)
          ;; emit t0 .. tn: the accepted drafts and the target's next token
          (loop for id in (subseq samples 0 (1+ n))
                do (when (or (member id eos-ids) (>= count max-tokens))
                     (return-from generation))
                   (incf count)
                   (funcall function id))
          ;; keep what both caches share with the accepted text
          (rewind-cache target-cache (+ target-offset (length pending) n))
          (let ((next (nth n samples)))
            (setf pending (list next))
            (if (< n k)
                (progn (rewind-cache draft-cache (+ draft-offset (length draft-pending) n))
                       (setf draft-pending (list next)))
                ;; every draft accepted: the draft never saw its own last one
                (setf draft-pending (list (car (last drafts)) next))))
          (when (>= count max-tokens) (return-from generation)))))
    (values count (or prompt-time 0)
            (/ (- (get-internal-real-time) start) internal-time-units-per-second)
            (if (plusp proposed) (/ accepted proposed) 0))))
