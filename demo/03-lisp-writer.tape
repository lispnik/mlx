Output demo/out/03-lisp-writer.mp4
Source demo/settings.tape
Hide
Type `demo/title.sh "3. A model that writes Lisp, runs it and repairs it" "Decoding follows the Lisp reader, so every reply is one readable form;" "parens the model miscounts are rebuilt from its indentation;" "the code runs in a sandboxed SBCL and failures go back to the model"`
Enter
Show
Sleep 5s
Hide
Type "clear"
Enter
Show
Type `bin/mlx-cl lisp -v 'Define (flatten tree) returning the atoms of a nested list in order' \`
Enter
Type `  -T "(equal (flatten '(1 (2 (3)) 4)) '(1 2 3 4))" -T "(null (flatten nil))"`
Sleep 300ms
Enter
Wait@180s /mlx ❯\s*$/
Sleep 4s
