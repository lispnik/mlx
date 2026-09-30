;;; emacs-init.el -- Emacs for the demo recording: SLY + mlx-complete  -*- lexical-binding: t -*-
(setq inhibit-startup-screen t
      inferior-lisp-program "sbcl"
      sly-net-coding-system 'utf-8-unix
      warning-minimum-level :error
      make-backup-files nil auto-save-default nil create-lockfiles nil)
(add-to-list 'load-path (or (getenv "SLY_DIR") (error "Set SLY_DIR to a SLY checkout")))
(add-to-list 'load-path (expand-file-name "emacs" default-directory))
(require 'sly)
(require 'mlx-complete)
(load-theme 'modus-vivendi t)
(menu-bar-mode -1)
(add-hook 'lisp-mode-hook #'mlx-complete-mode)
(defun demo-focus-file ()
  (switch-to-buffer (get-file-buffer "/tmp/shop.lisp"))
  (delete-other-windows))
(add-hook 'sly-connected-hook (lambda () (run-with-timer 1.0 nil #'demo-focus-file)
                                 (run-with-timer 1.5 nil #'mlx-complete-preload)))
(add-hook 'emacs-startup-hook (lambda () (save-window-excursion (sly))))
