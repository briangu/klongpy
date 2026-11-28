(def (sym require) (fn 3 (params ok kind message) (cond (sym ok) (num 1) (call (sym .err) (sym kind) (sym message)))))
(def (sym export) (edict (str "require") (sym require)))
