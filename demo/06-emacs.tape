Output demo/out/06-emacs.mp4
Source demo/settings.tape
Hide
Type `demo/title.sh "6. Completion in Emacs, from the live image" "A code model runs inside the Lisp that SLY is connected to. Its prompt holds" "what the image knows (your functions' lambda lists and docstrings); decoding" "follows the reader, and the result is compiled in the image to check it"`
Enter
Show
Sleep 5s
Hide
Type "clear"
Enter
Type "cp demo/shop.lisp /tmp/shop.lisp; emacs -nw -Q -l demo/emacs-init.el /tmp/shop.lisp"
Enter
Sleep 25s
Show
Sleep 1.5s

Ctrl+C
Ctrl+K
Sleep 4s
Escape
Type ">"
Type "(defun cart-total (items)"
Enter
Type `"The total price of ITEMS, with tax."`
Enter
Sleep 800ms
Ctrl+C
Tab
Sleep 4s
Tab
Sleep 3s
Enter
Enter
Type "(defun most-expensive (items)"
Enter
Ctrl+C
Tab
Sleep 5s
Tab
Sleep 4s
