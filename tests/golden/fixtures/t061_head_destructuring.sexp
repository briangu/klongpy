(def (sym f) (fn 2 (params v k) (locals p q) (destructure (p q) (sym v)) (d "*" (sym k) (d "+" (sym p) (sym q)))))
