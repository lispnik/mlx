;;;; ffi/library.lisp -- locate and load libmlxc

(in-package :mlx-ffi)

;; The raw bindings pass mlx-c's one-word handle structs as pointers and
;; pack optionals into a uint64.  That is exactly right for the AAPCS64
;; calling convention used on Apple Silicon -- the only platform MLX's
;; Metal backend supports -- but not guaranteed elsewhere.
#-(or arm64 aarch64)
(warn "mlx-ffi: bindings assume the AAPCS64 (arm64) calling convention.")

(cffi:define-foreign-library libmlxc
  (:darwin (:or "libmlxc.dylib"
                "/opt/homebrew/lib/libmlxc.dylib"
                "/usr/local/lib/libmlxc.dylib"))
  (:unix (:or "libmlxc.so" "/usr/local/lib/libmlxc.so"))
  (t (:default "libmlxc")))

(defun load-libmlxc ()
  "Load libmlxc.  $MLX_C_LIBRARY, if set, names the library file to use."
  (let ((override (uiop:getenv "MLX_C_LIBRARY")))
    (if (and override (plusp (length override)))
        (cffi:load-foreign-library override)
        (cffi:load-foreign-library 'libmlxc))))

(load-libmlxc)

(defconstant +trap-enable-bits+ #x1f00
  "Trap-enable bits of SB-VM:FLOATING-POINT-MODES on arm64.")

(defmacro with-float-traps-masked* (&body body)
  "Run BODY with all float traps masked.  MLX (Metal's allocator,
Accelerate) leaves IEEE exception flags set, which SBCL on arm64 turns into
Lisp errors after a foreign call returns.  Changing the modes is costly, so
when traps are already masked (e.g. in an enclosing call) BODY just runs."
  (let ((thunk (gensym "BODY")))
    `(flet ((,thunk () ,@body))
       (declare (dynamic-extent #',thunk) (inline ,thunk))
       (if (logtest (sb-vm:floating-point-modes) +trap-enable-bits+)
           (sb-int:with-float-traps-masked (:overflow :invalid :divide-by-zero :inexact :underflow)
             (,thunk))
           (,thunk)))))

(defmacro define-mlx-function ((c-name lisp-name) return-type &body doc-and-args)
  "Define %LISP-NAME, the raw foreign function C-NAME, and LISP-NAME, an
inline wrapper that calls it with float traps masked."
  (let* ((doc (when (stringp (first doc-and-args)) (first doc-and-args)))
         (args (if doc (rest doc-and-args) doc-and-args))
         (names (mapcar #'first args))
         (raw (intern (format nil "%~A" (symbol-name lisp-name)))))
    `(progn
       (cffi:defcfun (,c-name ,raw) ,return-type ,@args)
       (declaim (inline ,lisp-name))
       (defun ,lisp-name ,names
         ,@(when doc (list doc))
         (with-float-traps-masked* (,raw ,@names))))))

(export '(libmlxc load-libmlxc with-float-traps-masked* define-mlx-function))
