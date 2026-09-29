Output demo/out/02-llm.mp4
Source demo/settings.tape
Hide
Type `demo/title.sh "2. Language models, in pure Lisp" "Hugging Face tokenizers, Llama / Qwen / Gemma / Phi / mixture-of-experts models," "token-identical to mlx-lm and as fast"`
Enter
Show
Sleep 3.5s
Hide
Type "clear"
Enter
Show
Type `bin/mlx-cl generate -m mlx-community/Qwen3-0.6B-4bit --no-think -t 0 -n 90 -v "In two sentences: why do Lisp programmers love macros?"`
Sleep 300ms
Enter
Wait@120s /mlx ❯\s*$/
Sleep 4s
