#!/bin/sh

pkgs=$(dnf list --installed "texlive-*" | cut -d " " -f1 | rev | cut -d "." -f2- | rev)
pkgs=$(echo $pkgs | sed "s/ /-doc /g" | sed "s/$/-doc/")
#echo $pkgs
sudo dnf install --skip-broken $pkgs
