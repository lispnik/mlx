Output demo/out/04-symreg.mp4
Source demo/settings.tape
Hide
Type `demo/title.sh "4. Symbolic regression on the GPU" "Genetic programming over Lisp expressions: the whole population runs as one" "compiled GPU stack machine, and gradient descent tunes every constant at once"`
Enter
Show
Sleep 4.5s
Hide
Type "clear"
Enter
Show
Type "head -4 demo/gravity.csv"
Sleep 300ms
Enter
Sleep 2s
Type "bin/mlx-cl symreg -q --seed 2 demo/gravity.csv"
Sleep 300ms
Enter
Wait@120s /mlx ❯\s*$/
Sleep 5s
Type "clear"
Enter
Type "bin/mlx-cl symreg --seed 1 -g 25 -f '(+ (square x0) (* 2.5 (sin x1)))'"
Sleep 300ms
Enter
Wait@120s /mlx ❯\s*$/
Sleep 5s
