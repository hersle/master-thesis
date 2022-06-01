#!/bin/sh

convert () {
	if [ $# -ne 3 ]; then
		echo "syntax: convert PDF-FILE TARGET-PAPER-SIZE SCALE?"
		exit 1
	fi

	if [ $3 -eq 0 ]; then
		includepdfopts="pages=-, noautoscale, frame" # do not scale pages
	else
		includepdfopts="pages=-" # scale pages
	fi
	texsrc="
	\\documentclass[$2paper]{article}
	\\usepackage{pdfpages}
	\\begin{document}
	\\includepdf[$includepdfopts]{$1}
	\\end{document}
	"

	name="$(basename "$1" | rev | cut -d. -f2- | rev)-$2" # filename without extension (e.g. thesis-b5)
	echo $texsrc > /tmp/$name.tex

	pdflatex -halt-on-error -output-directory /tmp/ /tmp/$name.tex > /dev/null # shut up
	if [ ! -f /tmp/$name.pdf ]; then
		echo "FAILED"
		exit 1
	fi
	mv /tmp/$name.pdf $name.pdf
	echo $name.pdf # return filename
}

echo -n "Scaling $1 to $2...: "
newpdf=$(convert $1 $2 1)
echo $newpdf

if [ $3 ]; then
	echo -n "Placing $newpdf on $3...: "
	newnewpdf=$(convert $newpdf $3 0)
	echo "$newnewpdf"
fi
