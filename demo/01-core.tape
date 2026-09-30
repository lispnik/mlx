Output demo/out/01-core.mp4
Source demo/settings.tape
Hide
Type `demo/title.sh "1. Arrays on the GPU" "All 305 mlx-c operations, generated from its headers; autodiff, compile, #M literals"`
Enter
Show
Sleep 3s
Hide
Type "clear"
Enter
Show
Type "bin/mlx-cl info | head -4"
Sleep 300ms
Enter
Sleep 2s
Type "bin/mlx-cl eval '(mx:@ #M((1.0 2) (3 4)) #M((5 6) (7 8)))'"
Sleep 300ms
Enter
Sleep 2.5s
Type "bin/mlx-cl eval '(funcall (mx:grad (lambda (x) (mx:sum (mx:square x)))) #M(1.0 2.0 3.0))'"
Sleep 300ms
Enter
Sleep 2.5s
Type "bin/mlx-cl bench -n 4096 -i 10"
Sleep 300ms
Enter
Sleep 3s
