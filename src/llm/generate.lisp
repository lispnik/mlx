;;;; llm/generate.lisp -- chat templates, sampling and generation

(in-package :mlx.llm)

;;; ------------------------------------------------------------------
;;; Chat templates
;;;
;;; Hugging Face chat templates are Jinja programs.  Rather than embed a
;;; Jinja interpreter we recognise the common formats (ChatML: SmolLM,
;;; Qwen...; Llama 3) and extract a template's built-in default system
;;; prompt.  Other templates are an error; pass a raw prompt instead.

(defun unescape-jinja (text)
  "Resolve the \\n escapes of a Jinja string literal."
  (with-output-to-string (out)
    (loop with i = 0
          while (< i (length text))
          do (if (and (char= (char text i) #\\) (< (1+ i) (length text)) (char= (char text (1+ i)) #\n))
                 (progn (write-char #\Newline out) (incf i 2))
                 (progn (write-char (char text i) out) (incf i))))))

(defun template-default-system (template)
  "The literal default system prompt a ChatML template inserts, or NIL.
The marker may be written with a real newline or a \\n escape."
  (dolist (marker (list (format nil "<|im_start|>system~%") "<|im_start|>system\\n"))
    ;; the first occurrences usually splice in messages[0]; take the first
    ;; one followed by literal text
    (loop for start = (search marker template) then (search marker template :start2 (1+ start))
          while start
          do (let* ((text-start (+ start (length marker)))
                    (end (search "<|im_end|>" template :start2 text-start))
                    (text (and end (subseq template text-start end))))
               (when (and text (plusp (length text)) (not (find #\{ text)) (not (find #\' text))
                          (not (find #\+ text)))
                 (return-from template-default-system (unescape-jinja text)))))))

(defun today-string ()
  "Today's date as Llama 3 templates print it, e.g. \"28 Sep 2026\"."
  (multiple-value-bind (s m h day month year) (get-decoded-time)
    (declare (ignore s m h))
    (format nil "~2,'0D ~A ~D" day
            (nth (1- month) '("Jan" "Feb" "Mar" "Apr" "May" "Jun" "Jul" "Aug" "Sep" "Oct" "Nov" "Dec"))
            year)))

(defun apply-chat-template (tokenizer messages &key (add-generation-prompt t))
  "Render MESSAGES -- a list of (role . content), roles \"system\", \"user\"
or \"assistant\" -- as the model's chat prompt string.  ChatML (SmolLM,
Qwen), Llama 3, Gemma and Phi-3 formats are recognised."
  (let ((template (or (tokenizer-chat-template tokenizer) "")))
    (cond
      ((search "<|im_start|>" template)
       (with-output-to-string (s)
         (let ((default (template-default-system template)))
           (when (and default (not (equal (car (first messages)) "system")))
             (format s "<|im_start|>system~%~A<|im_end|>~%" default)))
         (loop for (role . content) in messages
               do (format s "<|im_start|>~A~%~A<|im_end|>~%" role content))
         (when add-generation-prompt (format s "<|im_start|>assistant~%"))))
      ((search "<|start_header_id|>" template)
       (with-output-to-string (s)
         (write-string "<|begin_of_text|>" s)
         (let ((system (and (equal (car (first messages)) "system") (cdr (pop messages)))))
           ;; Llama 3.1+ templates always open with a system turn carrying
           ;; the knowledge cutoff and today's date
           (when (or system (search "Cutting Knowledge Date" template))
             (format s "<|start_header_id|>system<|end_header_id|>~%~%")
             (when (search "Cutting Knowledge Date" template)
               (format s "Cutting Knowledge Date: December 2023~%Today Date: ~A~%~%" (today-string)))
             (format s "~@[~A~]<|eot_id|>" system)))
         (loop for (role . content) in messages
               do (format s "<|start_header_id|>~A<|end_header_id|>~%~%~A<|eot_id|>" role content))
         (when add-generation-prompt (format s "<|start_header_id|>assistant<|end_header_id|>~%~%"))))
      ((search "<start_of_turn>" template)
       ;; Gemma: the system prompt (if the template allows one) prefixes the
       ;; first user turn; assistant turns are "model"; content is trimmed
       (let* ((system (and (equal (car (first messages)) "system") (cdr (pop messages)))))
         (when (and system (search "System role not supported" template))
           (error "This model's chat template does not support a system prompt."))
         (with-output-to-string (s)
           (write-string (or (bos-string tokenizer) "") s)
           (loop for (role . content) in messages
                 for first = t then nil
                 do (format s "<start_of_turn>~A~%~@[~A~%~%~]~A<end_of_turn>~%"
                            (if (equal role "assistant") "model" role)
                            (and first system)
                            (string-trim '(#\Space #\Tab #\Newline #\Return) content)))
           (when add-generation-prompt (format s "<start_of_turn>model~%")))))
      ((search "<|end|>" template)
       ;; Phi-3
       (with-output-to-string (s)
         (loop for (role . content) in messages
               unless (and (equal role "system") (zerop (length content)))
                 do (format s "<|~A|>~%~A<|end|>~%" role content))
         (when add-generation-prompt (format s "<|assistant|>~%"))))
      (t (error "Unrecognised chat template; use a raw prompt.")))))

;;; ------------------------------------------------------------------
;;; Sampling

(defun make-sampler (&key (temperature 0.0) (top-p 1.0))
  "A function from logits (B vocab) to sampled token ids (B).  TEMPERATURE 0
is greedy; TOP-P < 1 samples from the smallest set of tokens whose
probability exceeds TOP-P (nucleus sampling)."
  (cond
    ((zerop temperature)
     (lambda (logits) (mx:argmax logits :axis -1)))
    ((< top-p 1.0)
     (lambda (logits)
       (let* ((probs (mx:softmax (mx:multiply logits (/ 1.0 temperature)) :axis -1))
              (order (mx:argsort probs :axis -1))
              (sorted (mx:take-along-axis probs order -1))
              (cumulative (mx:cumsum sorted -1))
              (kept (mx:where (mx:greater cumulative (- 1.0 top-p)) sorted 0))
              (choice (random:categorical (mx:log kept) :axis -1)))
         (mx:squeeze (mx:take-along-axis order (mx:expand-dims choice -1) -1) :axis -1))))
    (t
     (lambda (logits)
       (random:categorical (mx:multiply logits (/ 1.0 temperature)) :axis -1)))))

;;; ------------------------------------------------------------------
;;; Generation

(defun generate-tokens (model prompt-ids function &key (max-tokens 256) (sampler (make-sampler))
                                                       eos-ids cache (prefill-step-size 2048))
  "Generate up to MAX-TOKENS token ids after PROMPT-IDS, calling FUNCTION
with each id as soon as it is known; stop at any of EOS-IDS (not passed to
FUNCTION).  Returns (values token-count prompt-seconds generation-seconds).

Each step's computation is queued (ASYNC-EVAL) before waiting for the
previous token, overlapping Lisp work with the GPU."
  (let ((cache (or cache (make-cache model)))
        (start (get-internal-real-time))
        (prompt-time 0) (count 0))
    (flet ((next-token (input)
             ;; logits of the last position -> the next token; everything
             ;; but that token is freed as soon as the step is built
             (mx:with-scope ()
               (funcall sampler (mx:ref (funcall model input cache) t -1))))
           (seconds-since (t0) (/ (- (get-internal-real-time) t0) internal-time-units-per-second)))
      ;; As mlx-lm does: prefill all but the last prompt token (in chunks),
      ;; then step the last one.  The schedule affects bf16 rounding, so
      ;; matching it makes greedy output identical to mlx-lm's.
      (loop for rest = prompt-ids then (nthcdr (length chunk) rest)
            for chunk = (subseq rest 0 (min prefill-step-size (1- (length rest))))
            while (rest rest)
            do (mx:with-scope ()
                 (funcall model (mx:from-lisp (list chunk) :dtype :int32) cache)
                 (mx:eval (loop for c in cache collect (kv-cache-keys c) collect (kv-cache-values c))))
            finally (setf prompt-ids rest))
      (let ((next (next-token (mx:from-lisp (list prompt-ids) :dtype :int32))))
        (mx:async-eval next)
        (unwind-protect
             (loop
               ;; queue step n+1 on the GPU *before* blocking on token n
               ;; (speculatively: it is wasted if token n ends generation)
               (let* ((following (and (< count max-tokens)
                                      (let ((f (next-token (mx:reshape next '(1 1)))))
                                        (mx:async-eval f)
                                        f)))
                      (id (mx:item next)))
                 (when (zerop count)
                   (setf prompt-time (seconds-since start) start (get-internal-real-time)))
                 (when (or (member id eos-ids) (>= count max-tokens))
                   (when following (mx:free following))
                   (return))
                 (incf count)
                 (funcall function id)
                 (mx:free next)
                 (setf next following)))
          (mx:free next))))
    (values count prompt-time
            (/ (- (get-internal-real-time) start) internal-time-units-per-second))))

(defun add-bos-p (tokenizer text)
  "Whether to prepend BOS to TEXT: the tokenizer asks for it and the text
(e.g. from a chat template) does not already start with it."
  (and (bos-token tokenizer)
       (not (and (bos-string tokenizer)
                 (eql 0 (search (bos-string tokenizer) text))))))

(defun generate (model prompt &key (max-tokens 256) (temperature 0.0) (top-p 1.0) seed
                                   (chat t) system stream (verbose nil))
  "Generate text from MODEL (from LOAD-MODEL) for PROMPT.  With CHAT (the
default) PROMPT is a user message wrapped in the model's chat template
(SYSTEM overrides the system prompt); otherwise it is raw text.  When
STREAM is a stream, text is written to it as it is generated.  With
VERBOSE, timing is reported on *ERROR-OUTPUT*.  Returns the generated text."
  (let* ((tokenizer (or (model-tokenizer model) (error "MODEL has no tokenizer.")))
         (text (if chat
                   (apply-chat-template tokenizer (append (and system (list (cons "system" system)))
                                                          (list (cons "user" prompt))))
                   prompt))
         (ids (encode tokenizer text :add-bos (add-bos-p tokenizer text)))
         (decoder (make-stream-decoder tokenizer))
         (out (make-string-output-stream)))
    (when seed (random:seed seed))
    (multiple-value-bind (count prompt-seconds seconds)
        (generate-tokens model ids
                         (lambda (id)
                           (let ((piece (decode-step decoder id)))
                             (write-string piece out)
                             (when stream (write-string piece stream) (force-output stream))))
                         :max-tokens max-tokens
                         :sampler (make-sampler :temperature temperature :top-p top-p)
                         :eos-ids (eos-tokens tokenizer))
      (when verbose
        (when stream (fresh-line stream) (force-output stream))
        (format *error-output* "~&~D prompt tokens at ~,1F tok/s; ~D generated at ~,1F tok/s; peak memory ~,2F GB~%"
                (length ids) (/ (length ids) (max prompt-seconds 1e-6))
                count (/ count (max seconds 1e-6))
                (/ (mx:peak-memory) 1e9))))
    (get-output-stream-string out)))
