;;;; nn/lora.lisp -- low-rank adapters (mlx.nn.LoRALinear, as in mlx-lm)
;;;;
;;;; A LORA-LINEAR wraps a frozen LINEAR or QUANTIZED-LINEAR and adds a
;;;; trainable low-rank update:  y = linear(x) + scale * (x A) B,  with A
;;;; (in rank) random and B (rank out) zero, so training starts from the
;;;; original model.  Children are named as in mlx-lm ("linear", "lora_a",
;;;; "lora_b"), so adapters load in either.

(in-package :mlx.nn.impl)

(nn:defmodule nn:lora-linear () ((scale :initarg :scale :reader lora-scale)))

(defun linear-dims (linear)
  "(values input-dims output-dims) of a LINEAR or QUANTIZED-LINEAR."
  (let ((w (nn:child linear "weight")))
    (if (typep linear 'nn:quantized-linear)
        (values (* (mx:dim (nn:child linear "scales") 1) (slot-value linear 'group-size)) (mx:dim w 0))
        (values (mx:dim w 1) (mx:dim w 0)))))

(defun nn:lora-linear (linear &key (rank 8) (scale 20.0))
  "LINEAR (a LINEAR or QUANTIZED-LINEAR, frozen here) plus a trainable
update of rank RANK scaled by SCALE."
  (multiple-value-bind (in out) (linear-dims linear)
    (let ((m (make-instance 'nn:lora-linear :scale scale))
          (bound (/ 1.0 (sqrt in))))
      (nn:register m "linear" (nn:freeze linear))
      (nn:register m "lora_a" (random:uniform :low (- bound) :high bound :shape (list in rank)))
      (nn:register m "lora_b" (mx:zeros (list rank out)))
      m)))

(defmethod nn:forward ((m nn:lora-linear) &rest args)
  (destructuring-bind (x) args
    (let ((y (funcall (nn:child m "linear") x))
          (z (mx:matmul (mx:matmul x (nn:child m "lora_a")) (nn:child m "lora_b"))))
      (mx:add y (mx:astype (mx:multiply z (lora-scale m)) (mx:dtype x))))))

(defun nn:linear-to-lora-layers (modules &key (rank 8) (scale 20.0)
                                              (keys '("q_proj" "k_proj" "v_proj" "o_proj"
                                                      "gate_proj" "up_proj" "down_proj")))
  "Freeze MODULES (a module or a list of them, e.g. a model's last
transformer blocks) and replace every LINEAR or QUANTIZED-LINEAR child
named in KEYS, at any depth, by a LORA-LINEAR around it.  Only the
adapters remain trainable.  Returns the number of layers adapted."
  (let ((count 0)
        (keys (mapcar #'name-string keys)))
    (dolist (root (if (listp modules) modules (list modules)) count)
      (nn:freeze root)
      (dolist (m (nn:modules root))
        (loop for (name . value) in (%children m)
              when (and (member name keys :test #'string=)
                        (typep value '(or nn:linear nn:quantized-linear)))
                do (setf (nn:child m name) (nn:lora-linear value :rank rank :scale scale))
                   (incf count))))))
