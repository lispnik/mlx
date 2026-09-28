;;;; tests/package.lisp

(defpackage :mlx-tests
  (:use :cl :fiveam)
  (:local-nicknames (:mx :mlx) (:linalg :mlx.linalg) (:fft :mlx.fft)
                    (:random :mlx.random) (:fast :mlx.fast) (:dist :mlx.distributed)
                    (:nn :mlx.nn) (:optim :mlx.optimizers))
  (:export #:run-tests))

(in-package :mlx-tests)

(def-suite :mlx :description "All mlx tests.")

(defun run-tests ()
  "Run the whole suite; returns T when every test passed.
$MLX_CL_TEST_DEVICE=cpu runs everything on the CPU (as on a machine or CI
runner without Metal)."
  (let ((device (uiop:getenv "MLX_CL_TEST_DEVICE")))
    (when (and device (plusp (length device)))
      (mx:set-default-device (intern (string-upcase device) :keyword))))
  (format t "~&MLX ~A, default device ~(~A~), Metal ~:[unavailable~;available~]~%"
          (mx:version) (mx:device-type (mx:default-device)) (mx:metal-available-p))
  (let ((results (run :mlx)))
    (explain! results)
    (results-status results)))

(defun lisp (a)
  "A's values as nested lists (or a number)."
  (mx:to-lisp a :as :list))

(defun approx= (x y &optional (tol 1e-4))
  "Recursive numeric comparison of numbers, lists and Lisp arrays."
  (cond ((and (numberp x) (numberp y)) (<= (abs (- x y)) (* tol (max 1 (abs x) (abs y)))))
        ((and (arrayp x) (arrayp y) (not (stringp x)))
         (and (equal (array-dimensions x) (array-dimensions y))
              (loop for i below (array-total-size x)
                    always (approx= (row-major-aref x i) (row-major-aref y i) tol))))
        ((and (consp x) (consp y))
         (and (= (length x) (length y)) (every (lambda (a b) (approx= a b tol)) x y)))
        (t (equal x y))))

(defun close-to (array expected &optional (tol 1e-4))
  (approx= (lisp array) expected tol))

(defun temp-file (name)
  (merge-pathnames (format nil "mlx-cl-test-~A-~A" (random 1000000 (make-random-state t)) name)
                   (uiop:temporary-directory)))

(defmacro with-temp-file ((var name) &body body)
  `(let ((,var (temp-file ,name)))
     (unwind-protect (progn ,@body)
       (when (probe-file ,var) (delete-file ,var)))))
