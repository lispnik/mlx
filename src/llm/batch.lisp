;;;; batch.lisp -- generating several sequences at once
;;;;
;;;; The prompts are left-padded to one length and run as one (B L) batch.
;;;; The cache records each row's padding so attention never looks at it;
;;;; since RoPE only depends on relative positions, the shared offsets are
;;;; harmless.  A decoding step costs little more for B rows than for one
;;;; (it is bound by reading the weights), so a batch multiplies throughput.

(in-package :mlx.llm)

(defun generate-tokens-batch (model prompts &key (max-tokens 256) (sampler (make-sampler))
                                                 eos-ids (pad-id 0) (prefill-step-size 2048))
  "Generate up to MAX-TOKENS ids after each of PROMPTS (lists of ids) at
once, each row stopping at any of EOS-IDS.  SAMPLER maps logits (B vocab)
to ids (B).  Returns (values id-lists prompt-seconds generation-seconds)."
  (let* ((width (reduce #'max prompts :key #'length))
         (padding (mapcar (lambda (p) (- width (length p))) prompts))
         (padded (mapcar (lambda (p pad) (append (make-list pad :initial-element pad-id) p)) prompts padding))
         (cache (make-cache model :padding padding))
         (rows (length prompts))
         (outputs (make-array rows :initial-element '()))
         (done (make-array rows :initial-element nil))
         (start (get-internal-real-time))
         (prompt-time 0))
    (flet ((next-tokens (input)
             (mx:with-scope ()
               (mx:astype (funcall sampler (mx:ref (funcall model input cache) t -1)) :int32)))
           (seconds-since (t0) (/ (- (get-internal-real-time) t0) internal-time-units-per-second)))
      ;; prefill all but the last column, in chunks
      (loop for from = 0 then to
            for to = (min (1- width) (+ from prefill-step-size))
            while (< from (1- width))
            do (mx:with-scope ()
                 (funcall model (mx:from-lisp (mapcar (lambda (p) (subseq p from to)) padded) :dtype :int32)
                          cache)
                 (mx:eval (loop for c in cache collect (kv-cache-keys c) collect (kv-cache-values c)))))
      (let ((next (next-tokens (mx:from-lisp (mapcar #'last padded) :dtype :int32))))
        (mx:async-eval next)
        (unwind-protect
             (loop for step from 0
                   do (let* ((following (and (< (1+ step) max-tokens)
                                             ;; queue step n+1 before reading step n
                                             (let ((f (next-tokens (mx:reshape next (list rows 1)))))
                                               (mx:async-eval f)
                                               f)))
                             (ids (coerce (mx:to-lisp next) 'list)))
                        (when (zerop step)
                          (setf prompt-time (seconds-since start) start (get-internal-real-time)))
                        (loop for id in ids for i from 0
                              unless (aref done i)
                                do (if (member id eos-ids)
                                       (setf (aref done i) t)
                                       (push id (aref outputs i))))
                        (mx:free next)
                        (setf next following)
                        (when (or (null following) (every #'identity done))
                          (return))))
          (when next (mx:free next))
          (dolist (c cache) (when (kv-cache-padding c) (mx:free (kv-cache-padding c)) (return))))))
    (values (map 'list #'reverse outputs)
            prompt-time
            (/ (- (get-internal-real-time) start) internal-time-units-per-second))))

(defun generate-batch (model prompts &key (max-tokens 256) (temperature 0.0) (top-p 1.0) seed
                                          (chat t) system (verbose nil) (thinking t))
  "Generate text from MODEL for each of PROMPTS (strings) at once; like
GENERATE otherwise.  Returns a list of texts."
  (let* ((tokenizer (or (model-tokenizer model) (error "MODEL has no tokenizer.")))
         (id-lists (mapcar (lambda (prompt)
                             (let ((text (if chat
                                             (apply-chat-template tokenizer
                                                                  (append (and system (list (cons "system" system)))
                                                                          (list (cons "user" prompt)))
                                                                  :thinking thinking)
                                             prompt)))
                               (encode tokenizer text :add-bos (add-bos-p tokenizer text))))
                           prompts)))
    (when seed (random:seed seed))
    (multiple-value-bind (outputs prompt-seconds seconds)
        (generate-tokens-batch model id-lists :max-tokens max-tokens
                                              :sampler (make-sampler :temperature temperature :top-p top-p)
                                              :eos-ids (eos-tokens tokenizer)
                                              :pad-id (or (first (eos-tokens tokenizer)) 0))
      (when verbose
        (let ((generated (reduce #'+ outputs :key #'length))
              (prompted (reduce #'+ id-lists :key #'length)))
          (format *error-output* "~&~D prompts: ~D prompt tokens at ~,1F tok/s; ~D generated at ~,1F tok/s; peak memory ~,2F GB~%"
                  (length prompts) prompted (/ prompted (max prompt-seconds 1e-6))
                  generated (/ generated (max seconds 1e-6)) (/ (mx:peak-memory) 1e9))))
      (mapcar (lambda (ids) (decode tokenizer ids)) outputs))))
