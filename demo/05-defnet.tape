Output demo/out/05-defnet.mp4
Source demo/settings.tape
Hide
Type `demo/title.sh "5. defnet: shapes checked at compile time" "The macro infers every tensor shape as it expands: layers size themselves," "and shape mistakes are compile errors that name the form"`
Enter
Show
Sleep 4.5s
Hide
Type "clear"
Enter
Show
Type "sbcl --noinform"
Enter
Sleep 1s
Type `(asdf:load-system "mlx")`
Enter
Sleep 3s
Type "(in-package :mlx-user)"
Enter
Sleep 800ms
Type "(nn:defnet cnn ((x (batch 28 28 1)) &key (classes 10))"
Enter
Type "  (-> x (conv2d 16 3 :padding 1) relu (max-pool-2d 2)"
Enter
Type "        (conv2d 32 3 :padding 1) relu (max-pool-2d 2)"
Enter
Type "        flatten (linear 128) relu (dropout 0.25) (linear classes)))"
Enter
Sleep 1.5s
Type "(nn:net-summary 'cnn)"
Enter
Sleep 5s
Type "(mx:shape (funcall (cnn :classes 3) (mx:zeros '(8 28 28 1))))"
Enter
Sleep 2.5s
Type "(sb-ext:exit)"
Enter
Sleep 800ms
Type "clear"
Enter
Type "cat demo/bad-net.lisp"
Enter
Sleep 4s
Type `sbcl --non-interactive --eval '(asdf:load-system "mlx")' --eval '(compile-file "demo/bad-net.lisp")' 2>&1 | grep -A4 'caught ERROR'`
Sleep 300ms
Enter
Sleep 8s
