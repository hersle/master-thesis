#!/bin/sh

pdfjam --paper a4paper --no-tidy project.pdf "$1" -o project_cut.pdf
