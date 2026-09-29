# Makefile for mlx (Common Lisp bindings to mlx-c)

SBCL ?= sbcl
LISP = $(SBCL) --noinform --non-interactive

.PHONY: all deps generate check-generated test test-emacs cli demo clean

all: cli

deps:            ## fetch dependencies into ./ocicl
	ocicl install

generate:        ## regenerate bindings from the installed mlx-c headers
	$(SBCL) --script tools/generate.lisp $(MLX_C_INCLUDE)

check-generated: ## fail if the generated files are stale for the installed headers
	$(SBCL) --script tools/generate.lisp --check $(MLX_C_INCLUDE)

test:            ## run the FiveAM suites (MLX_CL_TEST_DEVICE=cpu forces the CPU;
                 ## MLX_CL_TEST_MODELS=1 adds tests against real model weights)
	$(LISP) --eval '(asdf:load-system "mlx/tests")' \
	        --eval '(uiop:quit (if (uiop:symbol-call :mlx-tests :run-tests) 0 1))'
	$(LISP) --eval '(asdf:load-system "mlx/llm-tests")' \
	        --eval '(uiop:quit (if (uiop:symbol-call :mlx-llm-tests :run-tests) 0 1))'
	$(LISP) --eval '(asdf:load-system "mlx/symreg-tests")' \
	        --eval '(uiop:quit (if (uiop:symbol-call :mlx-symreg-tests :run-tests) 0 1))'

test-emacs:      ## run the ERT tests of emacs/mlx-complete.el (needs emacs)
	emacs --batch -L emacs -l mlx-complete-tests -f ert-run-tests-batch-and-exit

demo: bin/mlx-cl  ## record demo/out/mlx-demo.mp4 (needs vhs, ffmpeg, emacs, $$SLY_DIR)
	mkdir -p demo/out
	for tape in demo/[0-9]*.tape; do vhs $$tape || exit 1; done
	cd demo/out && ls [0-9]*.mp4 | sed "s/.*/file '&'/" > list.txt && \
	  ffmpeg -loglevel error -y -f concat -safe 0 -i list.txt -c:v libx264 -crf 20 \
	         -pix_fmt yuv420p -movflags +faststart mlx-demo.mp4

cli: bin/mlx-cl  ## build the command-line driver

bin/mlx-cl: mlx.asd src/*.lisp src/*/*.lisp cli/*.lisp
	$(LISP) --eval '(asdf:make "mlx/cli")'

clean:
	rm -rf bin
	find . -name '*.fasl' -not -path './ocicl/*' -delete
