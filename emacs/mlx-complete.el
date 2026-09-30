;;; mlx-complete.el --- Local model completion for Common Lisp, via SLY or SLIME  -*- lexical-binding: t -*-

;; Author: Matthew Kennedy
;; URL: https://github.com/lispnik/mlx
;; Version: 0.1.0
;; Package-Requires: ((emacs "27.1"))
;; Keywords: lisp, tools, completion

;;; Commentary:

;; Completes the Common Lisp at point with a code model running inside
;; the connected Lisp image, on Apple silicon (mlx.llm:emacs-complete).
;; Because the model runs in the image, its prompt includes what the image
;; knows: the lambda lists and docstrings of the functions near point and
;; of the current package's definitions.  Decoding follows the Lisp reader,
;; so a completion keeps the form well formed, and the finished form is
;; compiled in the image to report undefined functions and variables.
;;
;; Setup, with SLY or SLIME connected to an SBCL that can load mlx/llm:
;;
;;   (require 'mlx-complete)
;;   (add-hook 'lisp-mode-hook #'mlx-complete-mode)
;;
;; Then C-c TAB (`mlx-complete-at-point') shows a completion as grey text:
;; TAB accepts it, anything else dismisses it.  With a prefix argument it
;; completes to the end of the line only.  The first use loads the model
;; (`mlx-complete-model', about 1 GB); `mlx-complete-preload' does that
;; ahead of time, e.g. from `sly-connected-hook'.

;;; Code:

(require 'cl-lib)

(declare-function sly-connected-p "sly")
(declare-function sly-eval-async "sly")
(declare-function sly-current-package "sly")
(declare-function slime-connected-p "slime")
(declare-function slime-eval-async "slime")
(declare-function slime-current-package "slime")

(defgroup mlx-complete nil
  "Local model completion for Common Lisp."
  :group 'lisp
  :prefix "mlx-complete-")

(defcustom mlx-complete-model "mlx-community/Qwen2.5-Coder-1.5B-4bit"
  "Hugging Face repo id or directory of a fill-in-the-middle code model."
  :type 'string)

(defcustom mlx-complete-context-chars 3000
  "How much text before the current top-level form to send."
  :type 'integer)

(defcustom mlx-complete-suffix-chars 1000
  "How much text after point to send."
  :type 'integer)

(defface mlx-complete-preview '((t :inherit shadow))
  "Face of a completion awaiting acceptance.")

(defvar-local mlx-complete--overlay nil
  "The overlay showing the pending completion.")

(defvar mlx-complete--request 0
  "Serial number of the latest request; older answers are ignored.")

;;; Talking to the Lisp

(defun mlx-complete--eval-async (form callback)
  "Evaluate FORM in the connected Lisp, then call CALLBACK with its value."
  (cond ((and (fboundp 'sly-connected-p) (sly-connected-p))
         (sly-eval-async form callback))
        ((and (fboundp 'slime-connected-p) (slime-connected-p))
         (slime-eval-async form callback))
        (t (user-error "mlx-complete: connect to a Lisp with SLY or SLIME first"))))

(defun mlx-complete--package ()
  "The Lisp package of the current buffer, as a string."
  (or (and (fboundp 'sly-current-package) (sly-current-package))
      (and (fboundp 'slime-current-package) (slime-current-package))
      "CL-USER"))

(defun mlx-complete--call-form (function &rest args)
  "A form calling mlx.llm's FUNCTION (a name) on ARGS.  It names mlx.llm's
symbols only at run time, loading mlx/llm first if the image lacks it."
  `(cl:let ((package (cl:or (cl:find-package "MLX.LLM")
                            (cl:progn (asdf:load-system "mlx/llm")
                                      (cl:find-package "MLX.LLM")))))
     (cl:setf (cl:symbol-value (cl:find-symbol "*COMPLETION-MODEL-NAME*" package)) ,mlx-complete-model)
     (cl:funcall (cl:find-symbol ,function package) ,@args)))

(defun mlx-complete--request-form (form-prefix context suffix package file mode)
  "The form asking the Lisp for a completion."
  (mlx-complete--call-form "EMACS-COMPLETE" form-prefix context suffix package file mode))

;;;###autoload
(defun mlx-complete-preload ()
  "Load mlx/llm and the model in the connected Lisp, in the background, so
the first completion is quick."
  (interactive)
  (message "mlx-complete: loading %s..." mlx-complete-model)
  (mlx-complete--eval-async (mlx-complete--call-form "EMACS-WARM-UP")
                            (lambda (_) (message "mlx-complete: ready"))))

(defun mlx-complete--form-start ()
  "Where the top-level form around point starts, or nil at top level."
  (car (nth 9 (syntax-ppss))))

;;; Showing and accepting

(defun mlx-complete-dismiss ()
  "Remove the pending completion."
  (interactive)
  (when (overlayp mlx-complete--overlay)
    (delete-overlay mlx-complete--overlay))
  (setq mlx-complete--overlay nil)
  (remove-hook 'pre-command-hook #'mlx-complete--before-command t))

(defun mlx-complete--before-command ()
  "Dismiss the pending completion unless the command accepts it."
  (unless (eq this-command 'mlx-complete-accept)
    (mlx-complete-dismiss)))

(defun mlx-complete-accept ()
  "Insert the pending completion."
  (interactive)
  (when (overlayp mlx-complete--overlay)
    (let ((text (overlay-get mlx-complete--overlay 'mlx-complete-text))
          (pos (overlay-start mlx-complete--overlay)))
      (mlx-complete-dismiss)
      (goto-char pos)
      (insert text))))

(defconst mlx-complete--accept-binding
  '(menu-item "" mlx-complete-accept :filter (lambda (cmd) (and mlx-complete--overlay cmd)))
  "TAB's binding in `mlx-complete-mode': accept, but only while a
completion is shown.  (A transient map would not do: the answer arrives
between commands, from the Lisp connection.)")

(defun mlx-complete--show (text complaints seconds &optional error)
  "Show TEXT at point as a pending completion."
  (mlx-complete-dismiss)
  (cond (error (message "mlx-complete: %s" error))
        ((string-empty-p text) (message "mlx-complete: nothing to add (%.1fs)" seconds))
        (t
         (let ((ov (make-overlay (point) (point) nil t nil)))
           (overlay-put ov 'mlx-complete-text text)
           (overlay-put ov 'after-string
                        (propertize text 'face 'mlx-complete-preview 'cursor t))
           (setq mlx-complete--overlay ov))
         (add-hook 'pre-command-hook #'mlx-complete--before-command nil t)
         (message "mlx-complete: TAB to accept (%.1fs)%s" seconds
                  (if complaints
                      (concat "; compiler: " (mapconcat #'identity complaints "; "))
                    "")))))

;;;###autoload
(defun mlx-complete-at-point (&optional line)
  "Complete the Common Lisp at point with the local model.
With prefix argument LINE, complete to the end of the line only."
  (interactive "P")
  (mlx-complete-dismiss)
  (let* ((point (point))
         (buffer (current-buffer))
         (tick (buffer-chars-modified-tick))
         (split (or (mlx-complete--form-start) point))
         (form-prefix (buffer-substring-no-properties split point))
         (context (buffer-substring-no-properties
                   (max (point-min) (- split mlx-complete-context-chars)) split))
         (suffix (buffer-substring-no-properties
                  point (min (point-max) (+ point mlx-complete-suffix-chars))))
         (request (cl-incf mlx-complete--request)))
    (message "mlx-complete: thinking...")
    (mlx-complete--eval-async
     (mlx-complete--request-form form-prefix context suffix (mlx-complete--package)
                                 (if buffer-file-name
                                     (file-name-nondirectory buffer-file-name)
                                   (buffer-name))
                                 (if line :line :form))
     (lambda (result)
       (when (and (buffer-live-p buffer) (= request mlx-complete--request))
         (with-current-buffer buffer
           ;; only if nothing changed while the model was thinking
           (when (and (= tick (buffer-chars-modified-tick)) (= point (point)))
             (apply #'mlx-complete--show result))))))))

;;;###autoload
(define-minor-mode mlx-complete-mode
  "Complete Common Lisp with a local model: \\[mlx-complete-at-point]."
  :lighter " mlx"
  :keymap (let ((map (make-sparse-keymap)))
            (define-key map (kbd "C-c TAB") #'mlx-complete-at-point)
            (define-key map (kbd "TAB") mlx-complete--accept-binding)
            (define-key map (kbd "<tab>") mlx-complete--accept-binding)
            map)
  (unless mlx-complete-mode (mlx-complete-dismiss)))

(provide 'mlx-complete)
;;; mlx-complete.el ends here
