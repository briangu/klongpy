(call (sym t) (str "{2*x}'!1000") (adv (verb (fn 1 (d "*" (num 2) (arg x)))) (adverbs "'") (m "!" (num 1000))) (d "*" (num 2) (m "!" (num 1000))))
(def (sym a) (m "!" (num 1000)))
(call (sym t) (str "a::!1000;2*a") (d "*" (num 2) (sym a)) (d "*" (num 2) (m "!" (num 1000))))
