;;;; llm/hub.lisp -- fetching models from the Hugging Face Hub

(in-package :mlx.llm)

(defun cache-root ()
  "Where downloaded models live: $MLX_CL_CACHE, else ~/.cache/mlx-cl/models/."
  (let ((env (uiop:getenv "MLX_CL_CACHE")))
    (uiop:ensure-directory-pathname
     (if (and env (plusp (length env)))
         env
         (merge-pathnames ".cache/mlx-cl/models/" (user-homedir-pathname))))))

(defun repo-id-p (source)
  "True if SOURCE looks like \"org/name\" rather than a local path."
  (and (stringp source)
       (= 1 (count #\/ source))
       (not (probe-file source))
       (not (find (char source 0) "/.~"))))

(defun curl (url &rest args)
  (let ((token (uiop:getenv "HF_TOKEN")))
    (uiop:run-program (append (list "curl" "-fsSL" "--retry" "3")
                              (when (and token (plusp (length token)))
                                (list "-H" (format nil "Authorization: Bearer ~A" token)))
                              args (list url))
                      :output :string :error-output :string)))

(defparameter *model-file-patterns* '("*.json" "*.safetensors" "tokenizer.model")
  "Repository files needed to run a model.")

(defun wanted-file-p (name)
  (and (not (find #\/ name))
       (some (lambda (pattern)
               (if (char= (char pattern 0) #\*)
                   (let ((suffix (subseq pattern 1)))
                     (and (>= (length name) (length suffix))
                          (string= suffix name :start2 (- (length name) (length suffix)))))
                   (string= pattern name)))
             *model-file-patterns*)))

(defun download-model (repo &key (revision "main") (verbose t))
  "Download the files of Hugging Face repository REPO needed to run it (config,
tokenizer, safetensors weights) into the cache, skipping files already
there.  Set $HF_TOKEN for gated models.  Returns the directory."
  (let* ((dir (merge-pathnames (format nil "~A/" repo) (cache-root)))
         (info (com.inuoe.jzon:parse
                (curl (format nil "https://huggingface.co/api/models/~A/revision/~A" repo revision))))
         (files (remove-if-not #'wanted-file-p
                               (map 'list (lambda (s) (json-get s "rfilename")) (json-get info "siblings")))))
    (ensure-directories-exist dir)
    (dolist (file files dir)
      (let ((target (merge-pathnames file dir)))
        (unless (probe-file target)
          (when verbose (format *error-output* "~&Downloading ~A/~A~%" repo file))
          ;; download beside the target, then rename, so an interrupted
          ;; download is never mistaken for a complete one
          (let ((part (make-pathname :type (format nil "~@[~A.~]part" (pathname-type target))
                                     :defaults target)))
            (curl (format nil "https://huggingface.co/~A/resolve/~A/~A" repo revision file)
                  "-o" (uiop:native-namestring part))
            (rename-file part target)))))))

(defun resolve-model (source)
  "The local directory for SOURCE: a directory as given, or the cached
download of a Hugging Face repo id (fetched if absent)."
  (if (repo-id-p source)
      (let ((dir (merge-pathnames (format nil "~A/" source) (cache-root))))
        (if (probe-file (merge-pathnames "config.json" dir))
            dir
            (download-model source)))
      (let ((dir (uiop:ensure-directory-pathname source)))
        (unless (probe-file (merge-pathnames "config.json" dir))
          (error "No config.json in ~A." dir))
        dir)))
