#!/bin/bash
for wavn in *.wav;do
   pngn="${wavn%.*}.png"
   sox "$wavn" -n spectrogram -r -m -s -X 500 -Y 800 -o "$pngn"
   echo $wavn
done
