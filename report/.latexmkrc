@default_files = ("thesis.tex");
$ENV{openout_any} = "a"; # let latex write to any directories, including hidden ones starting with . (e.g. .cachetikz)
$pdflatex = 'lualatex --shell-escape --file-line-error %O %S'; # use xelatex (faster) or lualatex (better supported)
$pdf_mode = 1;
$out_dir = '.cachelatex/';
$compiling_cmd = '[ ! -f cover-optimized.pdf ] && ./optimize.sh cover.pdf cover-optimized.pdf';
$success_cmd = '[ ! -f thesis.pdf ] && ln -s .cachelatex/thesis.pdf thesis.pdf';
$clean_full_ext = "auxlock %R-figure-*.*";
