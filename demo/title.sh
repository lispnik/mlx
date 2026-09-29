#!/bin/sh
# title.sh HEADING [LINE...] -- a title card for the demo recordings
clear
printf '\n\n\n\n'
printf '    \033[1;38;5;183m%s\033[0m\n\n' "$1"
shift
for line in "$@"; do printf '    \033[38;5;250m%s\033[0m\n' "$line"; done
