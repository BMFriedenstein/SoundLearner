# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 21:08:27 2018

@author: Brandon
"""

import os
from PIL import Image
import csv
import traceback

directory = "data/"
output = "data/img/img"
bin_pixels = 3

image_count = 0
for infile in os.listdir(directory+"spec/"):
    filename = os.path.splitext(infile)[0]
    data_name = directory + filename + ".meta"
    out_name = output + filename + "_c.png"
    try:
        with open(r""+data_name) as f:
            data_list = []
            reader = csv.reader(f, delimiter=',', quotechar='"')
            for row in reader:
                data_list.append(row)
        frequencies = data_list[0]
        velocity = float(data_list[1][0])
        sustain = data_list[2]
        im = Image.open(directory + "spec/" + infile)
        rgb_img = Image.new("RGBA", im.size)   
        width, height = rgb_img.size   
        
        xdelta = int(width/len(frequencies))
        freq_i = 0
        for x in range(0, width):
            if(x % xdelta is 0 and x >= xdelta):
                freq_i += 1   
            y_start = height - (float(frequencies[freq_i])/2000)*height - bin_pixels/2
            y_end = height - (float(frequencies[freq_i])/2000)*height + bin_pixels/2
            b = 0
            if(sustain[x] =='1'):
                b=255
            for y in range(0, height):
                r = pix_val = im.getpixel((x, y))
                g = 0            
                if(y >y_start and y< y_end):
                   g = 255 * velocity
                new_color = (int(r),int(g),int(b),255)            
                rgb_img.putpixel( (x,y), new_color)
        rgb_img.save(out_name)
        print(str(image_count) + ". " +out_name)
        image_count += 1
    except Exception as e:
        print("There was an error with " + filename)
        traceback.print_exc()
        
