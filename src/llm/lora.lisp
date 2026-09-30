;;;; lora.lisp -- fine-tuning a language model with low-rank adapters
;;;;
;;;; APPLY-LORA freezes a model and adds LORA-LINEAR adapters to its last
;;;; transformer blocks (on a 4-bit model this is QLoRA: the quantized
;;;; weights stay as they are).  LORA-TRAIN trains them on chat examples
;;;; with the loss on the reply tokens only.  SAVE-ADAPTERS writes just the
;;;; adapters (a few MB); LOAD-ADAPTERS puts them on a freshly loaded model.

(in-package :mlx.llm)

(defun apply-lora (model &key (rank 8) (scale 20.0) (num-layers 16)
                              (keys '("q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj")))
  "Freeze MODEL and add adapters to the KEYS linear layers of its last
NUM-LAYERS blocks.  Returns MODEL."
  (let* ((blocks (nn:child (nn:child model :model) :layers))
         (tuned (last blocks (min num-layers (length blocks)))))
    (nn:freeze model)
    (nn:linear-to-lora-layers tuned :rank rank :scale scale :keys keys)
    (setf (getf (model-lora model) :config)
          (list :rank rank :scale scale :num-layers num-layers :keys keys))
    model))

(defun save-adapters (model file)
  "Write MODEL's trainable arrays (its adapters) to FILE (.safetensors),
with their settings in FILE's .sexp sibling.  Returns FILE."
  (mx:save-safetensors file (nn:flatten-parameters (nn:trainable-parameters model)))
  (with-open-file (out (make-pathname :type "sexp" :defaults file) :direction :output :if-exists :supersede)
    (with-standard-io-syntax (prin1 (getf (model-lora model) :config) out)))
  file)

(defun load-adapters (model file)
  "Add the adapters saved in FILE (by SAVE-ADAPTERS) to MODEL.  Returns MODEL."
  (let ((config (with-open-file (in (make-pathname :type "sexp" :defaults file))
                  (let ((*read-eval* nil)) (read in)))))
    (apply #'apply-lora model config)
    (nn:load-weights model file :strict nil)
    model))

(defun chat-example (tokenizer messages reply)
  "(prompt-ids . reply-ids) for a chat: MESSAGES as the prompt (with the
generation prompt), REPLY as the assistant's answer and end of turn."
  (let* ((prompt (apply-chat-template tokenizer messages :thinking nil))
         (full (apply-chat-template tokenizer (append messages (list (cons "assistant" reply)))
                                    :add-generation-prompt nil :thinking nil))
         (prompt-ids (encode tokenizer prompt :add-bos (add-bos-p tokenizer prompt)))
         (full-ids (encode tokenizer full :add-bos (add-bos-p tokenizer full))))
    (unless (equal prompt-ids (subseq full-ids 0 (min (length full-ids) (length prompt-ids))))
      (error "The chat template does not extend the prompt; cannot separate the reply."))
    (cons prompt-ids (subseq full-ids (length prompt-ids)))))

(defun batch-arrays (examples pad-id)
  "(values inputs targets mask) for EXAMPLES ((prompt . reply) id lists):
the sequences right-padded, targets shifted by one, and the mask 1 where
the target is a reply token."
  (let* ((sequences (mapcar (lambda (e) (append (car e) (cdr e))) examples))
         (width (1- (reduce #'max sequences :key #'length)))
         (inputs '()) (targets '()) (masks '()))
    (loop for (prompt . reply) in examples
          for sequence in sequences
          do (let ((padded (append sequence (make-list (- (1+ width) (length sequence)) :initial-element pad-id))))
               (push (subseq padded 0 width) inputs)
               (push (subseq padded 1) targets)
               ;; target i is token i+1: a reply token when i+1 >= (length prompt)
               (push (loop for i below width
                           collect (if (and (>= (1+ i) (length prompt)) (< (1+ i) (length sequence))) 1.0 0.0))
                     masks)))
    (values (mx:from-lisp (nreverse inputs) :dtype :int32)
            (mx:from-lisp (nreverse targets) :dtype :int32)
            (mx:from-lisp (nreverse masks) :dtype :float32))))

(defun lora-train (model examples &key (iterations 300) (batch-size 4) (learning-rate 1e-4)
                                       (max-length 768) (rank 8) (scale 20.0) (num-layers 16)
                                       (report-every 10) seed (stream *standard-output*))
  "Fine-tune MODEL on EXAMPLES, a list of (prompt-ids . reply-ids) (see
CHAT-EXAMPLE), adding adapters first (see APPLY-LORA) unless it has them.
Each of ITERATIONS steps trains on BATCH-SIZE random examples with Adam;
the loss is the cross-entropy of the reply tokens.  Examples longer than
MAX-LENGTH tokens are skipped.  Returns the list of reported losses."
  (let* ((*random-state* (if seed (sb-ext:seed-random-state seed) (make-random-state t)))
         (examples (coerce (remove-if (lambda (e) (> (+ (length (car e)) (length (cdr e))) max-length)) examples)
                           'vector))
         (pad-id (or (first (eos-tokens (model-tokenizer model))) 0))
         (losses '()) (window 0) (window-tokens 0) (start (get-internal-real-time)))
    (when (zerop (length examples)) (error "No examples of at most ~D tokens." max-length))
    (unless (getf (model-lora model) :config)
      (apply-lora model :rank rank :scale scale :num-layers num-layers))
    (nn:train-mode model t)
    (let* ((optimizer (optim:adam learning-rate))
           (loss (lambda (inputs targets mask)
                   (let ((per-token (nn:cross-entropy (funcall model inputs nil) targets :reduction :none)))
                     (mx:divide (mx:sum (mx:multiply (mx:astype per-token :float32) mask)) (mx:sum mask)))))
           (step (nn:value-and-grad model loss)))
      (format stream "~&~:D trainable parameters; ~D examples~%"
              (reduce #'+ (mapcar (lambda (p) (mx:size (cdr p)))
                                  (nn:flatten-parameters (nn:trainable-parameters model))))
              (length examples))
      (loop for iteration from 1 to iterations
            do (mx:with-scope ()
                 (let ((batch (loop repeat batch-size collect (aref examples (random (length examples))))))
                   (multiple-value-bind (inputs targets mask) (batch-arrays batch pad-id)
                     (multiple-value-bind (value grads) (funcall step inputs targets mask)
                       (optim:update optimizer model grads)
                       (mx:eval value (nn:trainable-parameters model) (optim:state optimizer))
                       (incf window (mx:item value))
                       (incf window-tokens (mx:item (mx:sum mask)))))))
               (when (zerop (mod iteration report-every))
                 (let ((mean (/ window report-every)))
                   (push mean losses)
                   (when stream
                     (format stream "~&iter ~4D  loss ~,4F  ~,1F reply tokens/s  peak ~,2F GB~%"
                             iteration mean
                             (/ window-tokens (max 1e-6 (/ (- (get-internal-real-time) start) internal-time-units-per-second)))
                             (/ (mx:peak-memory) 1e9))
                     (force-output stream))
                   (setf window 0 window-tokens 0 start (get-internal-real-time))))))
    (nn:train-mode model nil)
    (nreverse losses)))
