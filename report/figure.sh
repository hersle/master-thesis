#!/bin/sh

xdg-open $1.pdf
tail -f --lines -30 $1.log
