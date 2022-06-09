#!/bin/sh

if [ $# == 0 ]; then
	echo "usage: grammar.sh [textidote-options] [texfile1] [texfile2] [...]"
	exit 1
fi

java -jar ~/.local/share/textidote/textidote.jar \
--read-all \
--check en \
--dict .vimspell.utf-8.add \
--ignore "sh:c:noin" \
--output html \
$@ > /tmp/textidote.html

firefox /tmp/textidote.html
