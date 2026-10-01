#!/bin/sh
# Distillation pipeline, one heavy job at a time (the machine has 16 GB).
cd "$(dirname "$0")/.."
while pgrep -f lisp-writer-bench >/dev/null; do sleep 30; done
L() { sbcl --noinform --non-interactive --load "$1" --eval "$2"; }
L train/lisp-writer-lora.lisp '(lisp-writer-lora:make-data)' > train/make-data.log 2>&1
L train/lisp-writer-lora.lisp '(lisp-writer-lora:train)' > train/train.log 2>&1
L bench/lisp-writer.lisp '(lisp-writer-bench:run :model "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit" :configs (quote (:plain :loop)))' > train/bench-3b.log 2>&1
L bench/lisp-writer.lisp '(lisp-writer-bench:run :model "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit" :adapters (asdf:system-relative-pathname "mlx" "train/adapters/qwen2.5-coder-3b-lisp.safetensors") :configs (quote (:plain :loop)))' > train/bench-3b-lora.log 2>&1
echo done > train/pipeline.done
