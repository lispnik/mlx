;;;; train/lisp-train-tasks.sexp -- exercises for fine-tuning the Lisp writer
;;;;
;;;; Disjoint from bench/lisp-tasks.sexp, which is kept for evaluation.
;;;; Same format: (:name :task :tests).

(
 (:name "sum-list" :task "Define (sum-list list) returning the sum of the numbers in LIST."
  :tests ("(= (sum-list '(1 2 3)) 6)" "(= (sum-list nil) 0)"))
 (:name "product-list" :task "Define (product-list list) returning the product of the numbers in LIST; the product of no numbers is 1."
  :tests ("(= (product-list '(2 3 4)) 24)" "(= (product-list nil) 1)"))
 (:name "my-length" :task "Define (my-length list) counting the elements of LIST without using LENGTH."
  :tests ("(= (my-length '(a b c)) 3)" "(= (my-length nil) 0)"))
 (:name "my-reverse" :task "Define (my-reverse list) reversing LIST without using REVERSE or NREVERSE."
  :tests ("(equal (my-reverse '(1 2 3)) '(3 2 1))" "(null (my-reverse nil))"))
 (:name "last-element" :task "Define (last-element list) returning the last element of LIST, or NIL if it is empty."
  :tests ("(eq (last-element '(a b c)) 'c)" "(null (last-element nil))"))
 (:name "nth-from-end" :task "Define (nth-from-end n list) returning the element N places from the end of LIST, 0 being the last."
  :tests ("(eq (nth-from-end 0 '(a b c)) 'c)" "(eq (nth-from-end 2 '(a b c)) 'a)"))
 (:name "remove-duplicates-keep-first" :task "Define (dedupe list) removing repeated elements (compared with EQUAL), keeping each first occurrence in order."
  :tests ("(equal (dedupe '(a b a c b)) '(a b c))" "(equal (dedupe '(\"x\" \"x\")) '(\"x\"))"))
 (:name "count-if-even" :task "Define (count-evens list) counting the even integers in LIST."
  :tests ("(= (count-evens '(1 2 3 4 6)) 3)" "(= (count-evens nil) 0)"))
 (:name "max-of-list" :task "Define (max-of-list list) returning the largest number in the non-empty LIST without using MAX with APPLY or REDUCE."
  :tests ("(= (max-of-list '(3 9 2)) 9)" "(= (max-of-list '(-5)) -5)"))
 (:name "average" :task "Define (average list) returning the arithmetic mean of the numbers in the non-empty LIST."
  :tests ("(= (average '(1 2 3 4)) 5/2)" "(= (average '(10)) 10)"))
 (:name "squares" :task "Define (squares n) returning the list of squares of 1 through N."
  :tests ("(equal (squares 4) '(1 4 9 16))" "(null (squares 0))"))
 (:name "evens-below" :task "Define (evens-below n) returning the even numbers from 0 below N in increasing order."
  :tests ("(equal (evens-below 7) '(0 2 4 6))" "(null (evens-below 0))"))
 (:name "factorial" :task "Define (factorial n) for non-negative integers N."
  :tests ("(= (factorial 0) 1)" "(= (factorial 5) 120)" "(= (factorial 20) 2432902008176640000)"))
 (:name "triangular" :task "Define (triangular n) returning 1 + 2 + ... + N."
  :tests ("(= (triangular 4) 10)" "(= (triangular 0) 0)"))
 (:name "collatz-steps" :task "Define (collatz-steps n) counting the steps for the Collatz sequence from N (halve evens, 3n+1 odds) to reach 1."
  :tests ("(= (collatz-steps 1) 0)" "(= (collatz-steps 6) 8)" "(= (collatz-steps 27) 111)"))
 (:name "perfect-p" :task "Define (perfect-p n) true when the positive integer N equals the sum of its proper divisors."
  :tests ("(perfect-p 6)" "(perfect-p 28)" "(not (perfect-p 12))"))
 (:name "prime-p" :task "Define (prime-p n) true when the integer N is prime."
  :tests ("(prime-p 2)" "(prime-p 97)" "(not (prime-p 1))" "(not (prime-p 91))"))
 (:name "prime-factors" :task "Define (prime-factors n) returning the prime factors of the integer N > 1 in non-decreasing order, with repetition."
  :tests ("(equal (prime-factors 12) '(2 2 3))" "(equal (prime-factors 97) '(97))"))
 (:name "binary-string" :task "Define (binary-string n) returning the binary digits of the non-negative integer N as a string, without FORMAT's ~B."
  :tests ("(string= (binary-string 10) \"1010\")" "(string= (binary-string 0) \"0\")"))
 (:name "parse-binary" :task "Define (parse-binary string) converting a string of 0s and 1s to the integer it denotes, without PARSE-INTEGER's :RADIX."
  :tests ("(= (parse-binary \"1010\") 10)" "(= (parse-binary \"0\") 0)"))
 (:name "reverse-digits" :task "Define (reverse-digits n) returning the non-negative integer N with its decimal digits reversed."
  :tests ("(= (reverse-digits 1230) 321)" "(= (reverse-digits 5) 5)"))
 (:name "armstrong-p" :task "Define (armstrong-p n) true when N equals the sum of its decimal digits each raised to the number of digits."
  :tests ("(armstrong-p 153)" "(armstrong-p 9474)" "(not (armstrong-p 100))"))
 (:name "leap-year-p" :task "Define (leap-year-p year) using the Gregorian rules."
  :tests ("(leap-year-p 2000)" "(not (leap-year-p 1900))" "(leap-year-p 2024)" "(not (leap-year-p 2023))"))
 (:name "clamp" :task "Define (clamp x low high) returning X limited to the range LOW to HIGH."
  :tests ("(= (clamp 5 0 3) 3)" "(= (clamp -1 0 3) 0)" "(= (clamp 2 0 3) 2)"))
 (:name "sign-of" :task "Define (sign-of x) returning -1, 0 or 1 according to the sign of the real number X, without SIGNUM."
  :tests ("(= (sign-of -3.5) -1)" "(= (sign-of 0) 0)" "(= (sign-of 7) 1)"))
 (:name "string-reverse" :task "Define (string-reverse string) reversing STRING without REVERSE."
  :tests ("(string= (string-reverse \"abc\") \"cba\")" "(string= (string-reverse \"\") \"\")"))
 (:name "capitalize-words" :task "Define (capitalize-words string) upper-casing the first letter of each space-separated word and lower-casing the rest, without STRING-CAPITALIZE."
  :tests ("(string= (capitalize-words \"hello wORLD\") \"Hello World\")"))
 (:name "count-char" :task "Define (count-char char string) counting occurrences of CHAR in STRING."
  :tests ("(= (count-char #\\a \"banana\") 3)" "(= (count-char #\\z \"abc\") 0)"))
 (:name "remove-spaces" :task "Define (remove-spaces string) returning STRING without its space characters."
  :tests ("(string= (remove-spaces \"a b  c\") \"abc\")"))
 (:name "starts-with-p" :task "Define (starts-with-p prefix string) true when STRING begins with PREFIX."
  :tests ("(starts-with-p \"ab\" \"abc\")" "(not (starts-with-p \"b\" \"abc\"))" "(starts-with-p \"\" \"x\")" "(not (starts-with-p \"abcd\" \"abc\"))"))
 (:name "ends-with-p" :task "Define (ends-with-p suffix string) true when STRING ends with SUFFIX."
  :tests ("(ends-with-p \"bc\" \"abc\")" "(not (ends-with-p \"ab\" \"abc\"))"))
 (:name "repeat-string" :task "Define (repeat-string string n) returning STRING repeated N times."
  :tests ("(string= (repeat-string \"ab\" 3) \"ababab\")" "(string= (repeat-string \"x\" 0) \"\")"))
 (:name "pad-left" :task "Define (pad-left string width char) padding STRING on the left with CHAR to at least WIDTH characters."
  :tests ("(string= (pad-left \"7\" 3 #\\0) \"007\")" "(string= (pad-left \"1234\" 3 #\\0) \"1234\")"))
 (:name "trim-string" :task "Define (trim string) removing leading and trailing spaces from STRING, without STRING-TRIM."
  :tests ("(string= (trim \"  hi there  \") \"hi there\")" "(string= (trim \"   \") \"\")"))
 (:name "acronym" :task "Define (acronym string) returning the upper-case first letters of the space-separated words of STRING."
  :tests ("(string= (acronym \"portable network graphics\") \"PNG\")"))
 (:name "vowel-free" :task "Define (disemvowel string) removing the vowels a, e, i, o, u in either case from STRING."
  :tests ("(string= (disemvowel \"Programming Lisp\") \"Prgrmmng Lsp\")"))
 (:name "char-frequencies" :task "Define (char-frequencies string) returning a hash table (test EQL) mapping each character of STRING to its count."
  :tests ("(let ((h (char-frequencies \"hello\"))) (and (= (gethash #\\l h) 2) (= (gethash #\\h h) 1)))"))
 (:name "first-repeated-char" :task "Define (first-repeated-char string) returning the first character that occurs again later in STRING, or NIL."
  :tests ("(char= (first-repeated-char \"abcbad\") #\\a)" "(null (first-repeated-char \"abc\"))"))
 (:name "is-pangram" :task "Define (pangram-p string) true when STRING contains every letter a to z, ignoring case."
  :tests ("(pangram-p \"The quick brown fox jumps over the lazy dog\")" "(not (pangram-p \"hello\"))"))
 (:name "word-lengths" :task "Define (word-lengths string) returning the lengths of the space-separated words of STRING, in order."
  :tests ("(equal (word-lengths \"a bb ccc\") '(1 2 3))"))
 (:name "longest-word" :task "Define (longest-word string) returning the longest space-separated word of STRING; the first one on ties."
  :tests ("(string= (longest-word \"a bbb cc ddd\") \"bbb\")"))
 (:name "zip" :task "Define (zip a b) pairing the elements of lists A and B into a list of two-element lists, as long as the shorter."
  :tests ("(equal (zip '(1 2 3) '(a b)) '((1 a) (2 b)))"))
 (:name "unzip" :task "Define (unzip pairs) turning a list of two-element lists into a list of two lists."
  :tests ("(equal (unzip '((1 a) (2 b))) '((1 2) (a b)))" "(equal (unzip nil) '(nil nil))"))
 (:name "rotate-left" :task "Define (rotate-left list n) moving the first N elements of LIST to its end."
  :tests ("(equal (rotate-left '(1 2 3 4 5) 2) '(3 4 5 1 2))" "(equal (rotate-left '(1 2) 0) '(1 2))"))
 (:name "take-while" :task "Define (take-while predicate list) returning the longest prefix of LIST whose elements satisfy PREDICATE."
  :tests ("(equal (take-while #'evenp '(2 4 5 6)) '(2 4))" "(null (take-while #'evenp '(1 2)))"))
 (:name "drop-while" :task "Define (drop-while predicate list) removing the longest prefix of LIST whose elements satisfy PREDICATE."
  :tests ("(equal (drop-while #'evenp '(2 4 5 6)) '(5 6))" "(null (drop-while #'evenp '(2 4)))"))
 (:name "partition" :task "Define (partition predicate list) returning two values: the elements satisfying PREDICATE and the others, each in order."
  :tests ("(equal (multiple-value-list (partition #'oddp '(1 2 3 4))) '((1 3) (2 4)))"))
 (:name "frequencies-alist" :task "Define (frequencies list) returning an alist of (element . count) in order of first occurrence, elements compared with EQUAL."
  :tests ("(equal (frequencies '(a b a c a)) '((a . 3) (b . 1) (c . 1)))"))
 (:name "cumulative-sum" :task "Define (cumulative-sum list) returning the running totals of the numbers in LIST."
  :tests ("(equal (cumulative-sum '(1 2 3 4)) '(1 3 6 10))" "(null (cumulative-sum nil))"))
 (:name "pairwise-differences" :task "Define (differences list) returning the differences between consecutive numbers in LIST."
  :tests ("(equal (differences '(1 4 9 16)) '(3 5 7))" "(null (differences '(5)))"))
 (:name "insert-sorted" :task "Define (insert-sorted x list) inserting the number X into the ascending list LIST, keeping it sorted."
  :tests ("(equal (insert-sorted 3 '(1 2 4 5)) '(1 2 3 4 5))" "(equal (insert-sorted 0 nil) '(0))"))
 (:name "insertion-sort" :task "Define (insertion-sort list) sorting a list of numbers ascending by insertion sort, without SORT."
  :tests ("(equal (insertion-sort '(3 1 2)) '(1 2 3))" "(null (insertion-sort nil))"))
 (:name "quicksort" :task "Define (quicksort list) sorting a list of numbers ascending with quicksort, without SORT."
  :tests ("(equal (quicksort '(5 3 8 1 9 2)) '(1 2 3 5 8 9))"))
 (:name "sorted-p" :task "Define (sorted-p list) true when the numbers in LIST are in non-decreasing order."
  :tests ("(sorted-p '(1 2 2 5))" "(not (sorted-p '(2 1)))" "(sorted-p nil)"))
 (:name "list-index" :task "Define (list-index item list) returning the position of ITEM (compared with EQUAL) in LIST, or NIL, without POSITION."
  :tests ("(= (list-index 'c '(a b c)) 2)" "(null (list-index 'z '(a b)))"))
 (:name "replace-all" :task "Define (replace-all old new list) replacing every element EQUAL to OLD with NEW, at every depth of the nested list LIST."
  :tests ("(equal (replace-all 'a 'x '(a (b a) c)) '(x (b x) c))"))
 (:name "count-atoms" :task "Define (count-atoms tree) counting the non-NIL atoms in a nested list."
  :tests ("(= (count-atoms '(a (b c) ((d)))) 4)" "(= (count-atoms nil) 0)"))
 (:name "tree-sum" :task "Define (tree-sum tree) adding up all numbers anywhere in a nested list, ignoring other atoms."
  :tests ("(= (tree-sum '(1 (2 x (3)) 4)) 10)"))
 (:name "list-depth-max" :task "Define (max-depth-element tree) returning the atom nested most deeply in the nested list TREE (the first on ties)."
  :tests ("(eq (max-depth-element '(a (b (c)) d)) 'c)"))
 (:name "assoc-update" :task "Define (alist-update key value alist) returning a new alist where KEY's entry has VALUE, added at the front if absent; keys compared with EQUAL."
  :tests ("(equal (alist-update 'b 9 '((a . 1) (b . 2))) '((a . 1) (b . 9)))" "(equal (alist-update 'c 3 '((a . 1))) '((c . 3) (a . 1)))"))
 (:name "hash-keys" :task "Define (hash-keys table) returning the keys of the hash table TABLE as a list sorted with STRING< on their names; keys are symbols."
  :tests ("(let ((h (make-hash-table))) (setf (gethash 'b h) 1 (gethash 'a h) 2) (equal (hash-keys h) '(a b)))"))
 (:name "invert-alist" :task "Define (invert-alist alist) swapping keys and values in each pair of ALIST."
  :tests ("(equal (invert-alist '((a . 1) (b . 2))) '((1 . a) (2 . b)))"))
 (:name "counter-closure" :task "Define (make-counter &optional (start 0)) returning a function of no arguments that returns START+1, START+2, ... on successive calls."
  :tests ("(let ((c (make-counter))) (and (= (funcall c) 1) (= (funcall c) 2)))" "(= (funcall (make-counter 10)) 11)"))
 (:name "adder" :task "Define (make-adder n) returning a function that adds N to its argument."
  :tests ("(= (funcall (make-adder 5) 3) 8)"))
 (:name "apply-n-times" :task "Define (apply-n-times function n x) applying FUNCTION to X N times."
  :tests ("(= (apply-n-times #'1+ 5 0) 5)" "(= (apply-n-times (lambda (x) (* 2 x)) 3 1) 8)"))
 (:name "unless-macro" :task "Define a macro (my-unless test &body body) that evaluates BODY when TEST is false, returning its last value, without using UNLESS or WHEN."
  :tests ("(= (my-unless nil 1 2) 2)" "(null (my-unless t 1))"))
 (:name "repeat-macro" :task "Define a macro (repeat-times n &body body) that evaluates BODY N times and returns NIL."
  :tests ("(let ((x 0)) (repeat-times 3 (incf x)) (= x 3))"))
 (:name "my-or-macro" :task "Define a macro (my-or &rest forms) that behaves like OR for any number of forms: it evaluates each form at most once and returns the first true value."
  :tests ("(= (my-or nil 3 4) 3)" "(null (my-or))" "(let ((n 0)) (my-or (incf n) (incf n)) (= n 1))"))
 (:name "for-range-macro" :task "Define a macro (for-range (var start end) &body body) that runs BODY with VAR bound to each integer from START below END."
  :tests ("(let ((s 0)) (for-range (i 1 4) (incf s i)) (= s 6))"))
 (:name "point-class" :task "Define a CLOS class POINT with slots X and Y (initargs :x and :y, accessors POINT-X and POINT-Y) and a function (distance p q) returning the Euclidean distance between two points."
  :tests ("(= (distance (make-instance 'point :x 0 :y 0) (make-instance 'point :x 3 :y 4)) 5)"))
 (:name "account-class" :task "Define a CLOS class ACCOUNT with a BALANCE slot (initarg :balance, accessor BALANCE, default 0), and generic functions (deposit account amount) and (withdraw! account amount) that update the balance and return it."
  :tests ("(let ((a (make-instance 'account))) (deposit a 10) (= (withdraw! a 4) 6))"))
 (:name "animal-speak" :task "Define classes DOG and CAT and a generic function SPEAK returning the string \"Woof\" for dogs and \"Meow\" for cats."
  :tests ("(string= (speak (make-instance 'dog)) \"Woof\")" "(string= (speak (make-instance 'cat)) \"Meow\")"))
 (:name "struct-temperature" :task "Define a structure TEMPERATURE with a slot CELSIUS (constructor MAKE-TEMPERATURE with :celsius), and (to-fahrenheit temperature) returning the Fahrenheit value."
  :tests ("(= (to-fahrenheit (make-temperature :celsius 100)) 212)"))
 (:name "parse-int-safe" :task "Define (parse-int-or-nil string) returning the integer STRING denotes, or NIL when it is not one, by handling the error PARSE-INTEGER signals."
  :tests ("(= (parse-int-or-nil \"42\") 42)" "(null (parse-int-or-nil \"4x\"))"))
 (:name "retry-restart" :task "Define (ensure-positive x) that signals an error of type SIMPLE-ERROR when X is not positive and otherwise returns X, and (safe-ensure x) returning X, or 1 when ENSURE-POSITIVE signals."
  :tests ("(= (ensure-positive 3) 3)" "(= (safe-ensure -2) 1)" "(= (safe-ensure 5) 5)"))
 (:name "matrix-transpose" :task "Define (transpose-lists rows) transposing a matrix given as a list of equal-length lists."
  :tests ("(equal (transpose-lists '((1 2 3) (4 5 6))) '((1 4) (2 5) (3 6)))"))
 (:name "dot-product" :task "Define (dot a b) returning the dot product of two equal-length lists of numbers."
  :tests ("(= (dot '(1 2 3) '(4 5 6)) 32)" "(= (dot nil nil) 0)"))
 (:name "vector-sum" :task "Define (vector-sum vector) summing the elements of the numeric vector VECTOR with a LOOP ACROSS."
  :tests ("(= (vector-sum #(1 2 3)) 6)" "(= (vector-sum #()) 0)"))
 (:name "array-max-index" :task "Define (argmax vector) returning the index of the largest element of the non-empty vector VECTOR (the first on ties)."
  :tests ("(= (argmax #(3 9 2 9)) 1)" "(= (argmax #(5)) 0)"))
 (:name "identity-matrix" :task "Define (identity-matrix n) returning an N x N 2D array with 1 on the diagonal and 0 elsewhere."
  :tests ("(equalp (identity-matrix 2) #2A((1 0) (0 1)))"))
 (:name "bubble-count" :task "Define (count-inversions list) counting the pairs of positions i < j in LIST with element i greater than element j."
  :tests ("(= (count-inversions '(3 1 2)) 2)" "(= (count-inversions '(1 2 3)) 0)"))
)
