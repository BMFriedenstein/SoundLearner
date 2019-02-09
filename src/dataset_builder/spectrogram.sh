#!/bin/bash

for wavn in $1*.wav;do
   pngn="${wavn%.*}.png"
   sox "$wavn" -n spectrogram -r -m -s -x 512 -y 512 -o "$pngn" -q 249 -w Hamming -d 00:04
   echo $wavn
done
