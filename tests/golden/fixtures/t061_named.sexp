(def (sym f) (fn 4 (params a b c d) (d "+" (sym a) (d "+" (sym b) (d "+" (sym c) (sym d))))))
(call (sym f) (num 1) (num 2) (num 3) (num 4))
