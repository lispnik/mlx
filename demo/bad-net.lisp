(in-package :mlx-user)

;; the second residual branch changes the width: 64 -> 32
(nn:defnet decoder ((x (batch seq 64)) &key (heads 4))
  (-> x
      (residual (layer-norm) (attention heads :mask :causal))
      (residual (layer-norm) (linear 256) gelu (linear 32))))
