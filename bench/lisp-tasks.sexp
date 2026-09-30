;;;; bench/lisp-tasks.sexp -- Common Lisp exercises for the Lisp writer
;;;;
;;;; Each task: (:name "..." :task "the request" :tests ("form" ...)); every
;;;; test form must return true.  Mixed difficulty: list and string
;;;; processing, recursion, hash tables, macros, CLOS, conditions, LOOP.

(
 (:name "flatten"
  :task "Define (flatten tree) returning the atoms of a nested list in order."
  :tests ("(equal (flatten '(1 (2 (3)) 4)) '(1 2 3 4))" "(null (flatten nil))" "(equal (flatten '((a) b ((c d)))) '(a b c d))"))
 (:name "run-length-encode"
  :task "Define (run-length-encode list) that turns (a a b c c c) into ((a . 2) (b . 1) (c . 3))."
  :tests ("(equal (run-length-encode '(a a b c c c)) '((a . 2) (b . 1) (c . 3)))" "(null (run-length-encode nil))"))
 (:name "while-macro"
  :task "Define a macro (while test &body body) that runs BODY repeatedly while TEST is true."
  :tests ("(let ((i 0) (s 0)) (while (< i 5) (incf s i) (incf i)) (= s 10))" "(let ((n 0)) (while nil (incf n)) (= n 0))"))
 (:name "primes-below"
  :task "Define (primes-below n) returning the list of primes below n in increasing order, using a sieve."
  :tests ("(equal (primes-below 20) '(2 3 5 7 11 13 17 19))" "(null (primes-below 2))" "(= (length (primes-below 1000)) 168)"))
 (:name "word-frequencies"
  :task "Define (word-frequencies string) returning an alist of (word . count) sorted by descending count, then alphabetically; words are separated by single spaces."
  :tests ("(equal (word-frequencies \"b a b c a b\") '((\"b\" . 3) (\"a\" . 2) (\"c\" . 1)))"))
 (:name "matrix-multiply"
  :task "Define (matrix-multiply a b) for 2D arrays of numbers."
  :tests ("(equalp (matrix-multiply #2A((1 2) (3 4)) #2A((5 6) (7 8))) #2A((19 22) (43 50)))" "(equalp (matrix-multiply #2A((1 2 3)) #2A((1) (2) (3))) #2A((14)))"))
 (:name "roman"
  :task "Define (roman n) converting a positive integer to a Roman numeral string, without using FORMAT's ~@R."
  :tests ("(string= (roman 1994) \"MCMXCIV\")" "(string= (roman 4) \"IV\")" "(string= (roman 3888) \"MMMDCCCLXXXVIII\")"))
 (:name "balanced-p"
  :task "Define (balanced-p string) that is true when the (), [] and {} brackets in STRING are balanced; other characters are ignored."
  :tests ("(balanced-p \"{[()()]}\")" "(not (balanced-p \"([)]\"))" "(not (balanced-p \"((\"))" "(balanced-p \"a(b)c\")"))
 (:name "fibonacci"
  :task "Define (fib n) returning the Nth Fibonacci number, with (fib 0) = 0 and (fib 1) = 1, efficient for n up to 1000."
  :tests ("(= (fib 10) 55)" "(= (fib 0) 0)" "(= (fib 100) 354224848179261915075)"))
 (:name "palindrome-p"
  :task "Define (palindrome-p string) that is true when STRING reads the same backwards, ignoring case and non-alphanumeric characters."
  :tests ("(palindrome-p \"A man, a plan, a canal: Panama\")" "(not (palindrome-p \"hello\"))" "(palindrome-p \"\")"))
 (:name "group-by"
  :task "Define (group-by key list) returning a hash table (test EQUAL) mapping each (funcall key x) to the list of elements with that key, in their original order."
  :tests ("(let ((h (group-by #'evenp '(1 2 3 4 5)))) (and (equal (gethash t h) '(2 4)) (equal (gethash nil h) '(1 3 5))))"))
 (:name "split-string"
  :task "Define (split-string string delimiter) splitting STRING at each occurrence of the character DELIMITER, keeping empty pieces."
  :tests ("(equal (split-string \"a,b,,c\" #\\,) '(\"a\" \"b\" \"\" \"c\"))" "(equal (split-string \"abc\" #\\,) '(\"abc\"))"))
 (:name "join-strings"
  :task "Define (join-strings strings separator) concatenating STRINGS with the string SEPARATOR between them."
  :tests ("(string= (join-strings '(\"a\" \"b\" \"c\") \", \") \"a, b, c\")" "(string= (join-strings nil \"-\") \"\")"))
 (:name "binary-search"
  :task "Define (binary-search vector item) returning the index of ITEM in the sorted vector of integers VECTOR, or NIL."
  :tests ("(= (binary-search #(1 3 5 7 9 11) 7) 3)" "(null (binary-search #(1 3 5) 4))" "(null (binary-search #() 1))"))
 (:name "merge-sort"
  :task "Define (merge-sort list) returning a new list of the numbers in LIST in increasing order, implemented as a merge sort without calling SORT."
  :tests ("(equal (merge-sort '(5 2 9 1 5 6)) '(1 2 5 5 6 9))" "(null (merge-sort nil))"))
 (:name "permutations"
  :task "Define (permutations list) returning all permutations of LIST."
  :tests ("(= (length (permutations '(1 2 3 4))) 24)" "(equal (sort (mapcar (lambda (p) (format nil \"~{~A~}\" p)) (permutations '(1 2 3))) #'string<) '(\"123\" \"132\" \"213\" \"231\" \"312\" \"321\"))"))
 (:name "power-set"
  :task "Define (power-set list) returning the list of all subsets of LIST."
  :tests ("(= (length (power-set '(a b c))) 8)" "(member nil (power-set '(a b)))"))
 (:name "memoize"
  :task "Define (memoize function) returning a function of one argument that caches FUNCTION's results in a hash table (test EQUAL)."
  :tests ("(let* ((calls 0) (f (memoize (lambda (x) (incf calls) (* x x))))) (and (= (funcall f 3) 9) (= (funcall f 3) 9) (= calls 1)))"))
 (:name "compose"
  :task "Define (compose &rest functions) returning the composition of FUNCTIONS, applied right to left; with none, the identity."
  :tests ("(= (funcall (compose #'1+ (lambda (x) (* 2 x))) 5) 11)" "(= (funcall (compose) 7) 7)"))
 (:name "swap-macro"
  :task "Define a macro (swap! a b) that exchanges the values of the two places A and B, evaluating each place's subforms only once."
  :tests ("(let ((x 1) (y 2)) (swap! x y) (and (= x 2) (= y 1)))" "(let ((v (vector 1 2))) (swap! (aref v 0) (aref v 1)) (equalp v #(2 1)))"))
 (:name "with-timing"
  :task "Define a macro (with-counter (var) &body body) that binds VAR to a function of no arguments which returns 1, 2, 3, ... on successive calls within BODY, and returns BODY's value."
  :tests ("(equal (with-counter (next) (list (funcall next) (funcall next) (funcall next))) '(1 2 3))"))
 (:name "stack-class"
  :task "Define a CLOS class STACK with generic functions (stack-push stack item), (stack-pop stack) returning the item, and (stack-empty-p stack). Make instances with (make-instance 'stack)."
  :tests ("(let ((s (make-instance 'stack))) (stack-push s 1) (stack-push s 2) (and (= (stack-pop s) 2) (= (stack-pop s) 1) (stack-empty-p s)))"))
 (:name "shapes-area"
  :task "Define classes CIRCLE (slot RADIUS, initarg :radius) and RECTANGLE (slots WIDTH and HEIGHT, initargs :width and :height), and a generic function AREA with a method for each."
  :tests ("(< (abs (- (area (make-instance 'circle :radius 1)) pi)) 1e-6)" "(= (area (make-instance 'rectangle :width 2 :height 3)) 6)"))
 (:name "safe-divide"
  :task "Define (safe-divide a b) returning A divided by B, or the keyword :DIVISION-BY-ZERO when B is zero, by handling the DIVISION-BY-ZERO condition."
  :tests ("(= (safe-divide 6 3) 2)" "(eq (safe-divide 1 0) :division-by-zero)"))
 (:name "custom-condition"
  :task "Define a condition INSUFFICIENT-FUNDS (a subclass of ERROR) with a slot for the amount needed (initarg :needed, reader NEEDED), and (withdraw balance amount) returning the new balance or signalling INSUFFICIENT-FUNDS with the shortfall."
  :tests ("(= (withdraw 10 3) 7)" "(= (handler-case (withdraw 5 8) (insufficient-funds (c) (needed c))) 3)"))
 (:name "count-vowels"
  :task "Define (count-vowels string) counting the vowels a, e, i, o, u in STRING, in either case."
  :tests ("(= (count-vowels \"Hello World\") 3)" "(= (count-vowels \"xyz\") 0)" "(= (count-vowels \"AEIOU\") 5)"))
 (:name "caesar"
  :task "Define (caesar string shift) shifting each letter of STRING by SHIFT places in the alphabet, wrapping around and keeping case; other characters are unchanged."
  :tests ("(string= (caesar \"Hello, World!\" 3) \"Khoor, Zruog!\")" "(string= (caesar \"abc\" -1) \"zab\")"))
 (:name "anagram-p"
  :task "Define (anagram-p a b) true when strings A and B are anagrams of each other, ignoring case and spaces."
  :tests ("(anagram-p \"Listen\" \"Silent\")" "(anagram-p \"a gentleman\" \"elegant man\")" "(not (anagram-p \"abc\" \"abd\"))"))
 (:name "deep-reverse"
  :task "Define (deep-reverse tree) reversing a list and, recursively, every list inside it."
  :tests ("(equal (deep-reverse '(1 (2 3) (4 (5 6)))) '(((6 5) 4) (3 2) 1))" "(null (deep-reverse nil))"))
 (:name "tree-depth"
  :task "Define (tree-depth tree) returning the nesting depth of a list: an atom has depth 0, a flat list depth 1."
  :tests ("(= (tree-depth 'a) 0)" "(= (tree-depth '(a b)) 1)" "(= (tree-depth '(a (b (c)))) 3)"))
 (:name "range"
  :task "Define (range start end &optional (step 1)) returning the list of numbers from START below END by STEP; STEP may be negative, counting down to above END."
  :tests ("(equal (range 0 5) '(0 1 2 3 4))" "(equal (range 0 10 3) '(0 3 6 9))" "(equal (range 5 0 -2) '(5 3 1))"))
 (:name "chunk"
  :task "Define (chunk list n) splitting LIST into consecutive sublists of length N (the last may be shorter)."
  :tests ("(equal (chunk '(1 2 3 4 5) 2) '((1 2) (3 4) (5)))" "(null (chunk nil 3))"))
 (:name "interleave"
  :task "Define (interleave a b) alternating elements of lists A and B, then appending the rest of the longer one."
  :tests ("(equal (interleave '(1 2 3) '(a b)) '(1 a 2 b 3))" "(equal (interleave nil '(x)) '(x))"))
 (:name "gcd-lcm"
  :task "Define (my-gcd a b) and (my-lcm a b) for positive integers without using the built-in GCD or LCM."
  :tests ("(= (my-gcd 48 18) 6)" "(= (my-lcm 4 6) 12)" "(= (my-gcd 7 13) 1)"))
 (:name "digits-sum"
  :task "Define (digit-sum n) returning the sum of the decimal digits of the non-negative integer N."
  :tests ("(= (digit-sum 12345) 15)" "(= (digit-sum 0) 0)" "(= (digit-sum (expt 2 100)) 115)"))
 (:name "queue-closure"
  :task "Define (make-queue) returning two values: an ENQUEUE function of one argument and a DEQUEUE function of none, sharing a first-in first-out queue; DEQUEUE returns NIL when it is empty."
  :tests ("(multiple-value-bind (enq deq) (make-queue) (funcall enq 1) (funcall enq 2) (and (= (funcall deq) 1) (= (funcall deq) 2) (null (funcall deq))))"))
 (:name "alist-to-plist"
  :task "Define (alist-to-plist alist) converting ((a . 1) (b . 2)) into (a 1 b 2), and (plist-to-alist plist) doing the reverse."
  :tests ("(equal (alist-to-plist '((a . 1) (b . 2))) '(a 1 b 2))" "(equal (plist-to-alist '(:x 1 :y 2)) '((:x . 1) (:y . 2)))"))
 (:name "most-frequent"
  :task "Define (most-frequent list) returning the element occurring most often in LIST (compared with EQUAL), and its count as a second value; ties go to the element seen first."
  :tests ("(equal (multiple-value-list (most-frequent '(a b a c b a))) '(a 3))" "(eq (most-frequent '(x y)) 'x)"))
 (:name "string-compress"
  :task "Define (compress string) replacing each run of a repeated character by the character followed by the run length, e.g. \"aaabcc\" becomes \"a3b1c2\"."
  :tests ("(string= (compress \"aaabcc\") \"a3b1c2\")" "(string= (compress \"\") \"\")"))
 (:name "infix-eval"
  :task "Define (eval-infix expr) evaluating an arithmetic expression written as a nested infix list, e.g. (1 + (2 * 3)), with operators + - * / and numbers; each list has exactly three elements."
  :tests ("(= (eval-infix '(1 + (2 * 3))) 7)" "(= (eval-infix '((10 - 4) / 2)) 3)" "(= (eval-infix 5) 5)"))
)
