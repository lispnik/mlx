;;;; tests/io.lisp -- npy, safetensors, gguf, in-memory, function export

(in-package :mlx-tests)

(def-suite :mlx.io :in :mlx)
(in-suite :mlx.io)

(test npy-file
  (with-temp-file (f "a.npy")
    (mx:save f (mx:from-lisp '((1.5 2.5) (3.5 4.5))))
    (let ((a (mx:load f)))
      (is (eq :float32 (mx:dtype a)))
      (is (equal '((1.5 2.5) (3.5 4.5)) (lisp a))))))

(test safetensors-file
  (with-temp-file (f "t.safetensors")
    (mx:save-safetensors f (list :w (mx:arange 3.0) :b (mx:ones '(2) :dtype :float16))
                         :metadata '(("format" . "mlx") ("epoch" . 3)))
    (multiple-value-bind (arrays metadata) (mx:load f)
      (is (= 2 (hash-table-count arrays)))
      (is (close-to (gethash "w" arrays) '(0 1 2)))
      (is (eq :float16 (mx:dtype (gethash "b" arrays))))
      (is (equal "mlx" (gethash "format" metadata)))
      (is (equal "3" (gethash "epoch" metadata))))))

(test gguf-file
  (with-temp-file (f "m.gguf")
    (mx:save-gguf f (list :w (mx:arange 4.0))
                  :metadata '(("name" . "test") ("tags" "a" "b") ("scale" . 2.0)))
    (multiple-value-bind (arrays gguf) (mx:load f)
      (is (close-to (gethash "w" arrays) '(0 1 2 3)))
      (is (equal "test" (mx:gguf-metadata gguf "name")))
      (is (equal '("a" "b") (mx:gguf-metadata gguf "tags")))
      (is (close-to (mx:gguf-metadata gguf "scale") 2))
      (is (null (mx:gguf-metadata gguf "missing"))))))

(test octets
  (let ((bytes (mx:save-to-octets (mx:from-lisp '(1 2 3)))))
    (is (typep bytes '(vector (unsigned-byte 8))))
    (is (equal '(1 2 3) (lisp (mx:load-from-octets bytes)))))
  (multiple-value-bind (arrays metadata)
      (mx:load-safetensors-from-octets
       (mx:save-safetensors-to-octets (list :x (mx:ones '(2 2))) :metadata '(("k" . "v"))))
    (is (close-to (gethash "x" arrays) '((1 1) (1 1))))
    (is (equal "v" (gethash "k" metadata)))))

(test missing-file-errors
  (signals mx:mlx-error (mx:load "/nonexistent/nope.npy")))

(test function-export
  (with-temp-file (f "fn.mlxfn")
    (mx:export-function f (lambda (x y) (mx:add (mx:multiply x 2) y)) (mx:ones '(2)) (mx:ones '(2)))
    (let ((g (mx:import-function f)))
      (is (close-to (first (funcall g (mx:from-lisp '(1.0 2.0)) (mx:from-lisp '(10.0 20.0))))
                    '(12 24))))))

(test function-export-kwargs
  (with-temp-file (f "kw.mlxfn")
    (mx:export-function f (lambda (x &key bias) (mx:add x bias)) (mx:ones '(2)) :bias (mx:ones '(2)))
    (let ((g (mx:import-function f)))
      (is (close-to (first (funcall g (mx:from-lisp '(1.0 2.0)) :bias (mx:from-lisp '(5.0 5.0))))
                    '(6 7))))))

(test function-exporter-multiple-traces
  (with-temp-file (f "multi.mlxfn")
    (mx:with-function-exporter (record f (lambda (x) (mx:multiply x 3)))
      (record (mx:ones '(2)))
      (record (mx:ones '(3))))
    (let ((g (mx:import-function f)))
      (is (close-to (first (funcall g (mx:ones '(2)))) '(3 3)))
      (is (close-to (first (funcall g (mx:ones '(3)))) '(3 3 3))))))
