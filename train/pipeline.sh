#!/bin/sh
# Measurements and distillation, one heavy job at a time (the machine has
# 16 GB).  Every step resumes from what it already recorded, so rerunning
# this script continues an interrupted run.
cd "$(dirname "$0")/.."
L() { sbcl --noinform --non-interactive --load "$1" --eval "$2"; }
STUDENT=mlx-community/Qwen2.5-Coder-3B-Instruct-4bit
ADAPTERS=train/adapters/qwen2.5-coder-3b-lisp.safetensors
# 1. does the richer feedback help the repair loop? (the tasks the first try misses)
L bench/lisp-writer.lisp '(lisp-writer-bench:run :configs (quote (:loop-bare :loop)) :tasks (with-open-file (in "bench/repair-tasks.sexp") (read in)))' > bench/run-diag.log 2>&1
# 2. the 7B writer solves the training exercises
L train/lisp-writer-lora.lisp '(lisp-writer-lora:make-data)' >> train/make-data.log 2>&1
# 3. adapters for the 3B model
[ -f "$ADAPTERS" ] || L train/lisp-writer-lora.lisp '(lisp-writer-lora:train)' > train/train.log 2>&1
# 4. base 3B against fine-tuned 3B on the benchmark
L bench/lisp-writer.lisp "(lisp-writer-bench:run :model \"$STUDENT\" :configs (quote (:plain :loop)))" > train/bench-3b.log 2>&1
L bench/lisp-writer.lisp "(lisp-writer-bench:run :model \"$STUDENT\" :adapters (asdf:system-relative-pathname \"mlx\" \"$ADAPTERS\") :configs (quote (:plain :loop)))" > train/bench-3b-lora.log 2>&1
echo done > train/pipeline.done
