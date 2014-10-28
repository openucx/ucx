;;
;; License goes here
;;

;;
;; Usage:
;; add our tweaks to c-mode
;; The basic indentation
;;   (add-hook 'c-mode-hook 'tc-indent-config)
;;
;; Warn when we're getting close to exceeding line length
;;  (add-hook 'c-mode-hook 'tc-mark-80-column-rule)
;;
;; On-the-fly indentation of lines
;;   (add-hook 'c-mode-hook 'electric-indent-mode)
;;
;; Indent opened lines immediately rather than TABbing across
;;   (add-hook 'c-mode-hook 'tc-c-nl-indent)
;;

(require 'cc-mode)
(require 'whitespace)

(defun tc-mark-80-column-rule ()
  "Highlight lines that are about to get too long"
  (setq whitespace-line-column 78)
  (setq whitespace-style '(face empty tabs lines-tail trailing))
  (whitespace-mode)
  )

(defun tc-indent-config ()
  "Set the required layout"
  (setq c-basic-offset 4
	tab-width 4
	indent-tabs-mode nil)
  )

;; add our tweaks to c-mode
(defun tc-c-nl-indent ()
  "Make newline also immediately indent next line"
  (define-key c-mode-base-map (kbd "RET") 'newline-and-indent)
  )

;; Other people can now require me
(provide 'ucx-style)
