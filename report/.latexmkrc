$ENV{openout_any} = a; # let latex write to any directories, including hidden ones starting with . (e.g. .cache_tikz)
$pdflatex = 'lualatex --shell-escape --file-line-error %O %S';
$pdf_mode = 1;
