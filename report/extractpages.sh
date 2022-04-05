#!/bin/sh

pdfjam --paper a4paper --no-tidy thesis.pdf "$1" -o thesis-cut.pdf
