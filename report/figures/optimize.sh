#!/bin/sh

name=$(echo "$1" | rev | cut -d. -f2 | rev)
 ext=$(echo "$1" | rev | cut -d. -f1 | rev)
outfile="$name-optimized.$ext"

du -bh $1
cp $1 $outfile
mogrify -quality "$2" $outfile
du -bh $outfile
