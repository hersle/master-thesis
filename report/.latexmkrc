$ENV{openout_any} = a; # let latex write to any directories, including hidden ones starting with . (e.g. .cachetikz)
$pdflatex = 'xelatex --shell-escape --file-line-error %O %S'; # use xelatex (faster) or lualatex (better supported)
$pdf_mode = 1;
