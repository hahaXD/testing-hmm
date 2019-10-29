(TeX-add-style-hook
 "thmm"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "amsmath"
    "subcaption")
   (TeX-add-symbols
    "pr"
    "esum")
   (LaTeX-add-labels
    "fig:length-8"
    "fig:length-11"
    "fig:exp"))
 :latex)

