(defpackage :shop (:use :cl))
(in-package :shop)

(defstruct item name price quantity)

(defun item-total (item &key (tax-rate 0.0))
  "Price times quantity, plus tax."
  (* (item-price item) (item-quantity item) (+ 1 tax-rate)))

