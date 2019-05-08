import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import math
import cv2
import brotli
from PIL import Image, ImageDraw
from io import BytesIO,SEEK_SET

import gc

#convert strokes to an image
min_color        = 10 #so that we can see the line
max_color        = 255-min_color #symmetri
max_stroke_coord = 255

def strokes2image(str_strokes:str, size:int=256, lw:int=6, linetype=cv2.LINE_8):
    stroke_scale= size / (max_stroke_coord+1)
    #scale coordinates to image size
    strokes     = [ [np.transpose((0.5+stroke_scale*np.asarray(s, dtype=np.float))).astype(dtype=np.int32)] 
                   for s in json.loads(str_strokes)]
    
    #scale color to the stroke index (alias for time)            
    scale_color = max_color//len(strokes)
    colors      = [ min_color + scale_color*i for i in range(len(strokes)) ]
    
    img         = np.zeros((size, size), np.uint8)
    for color, s in zip(colors,strokes):
        cv2.polylines(img, s, False, color, lw, linetype)
        #cv2.polylines(img, s, isClosed=False, color=color, thickness=lw)
        
    return img

def convert_strokes2image(drawings, size, line_width, linetype, compressed_drawings):
    "convert strokes to images"
    ims = []
    for drawing in drawings : 
        if compressed_drawings:
            im = strokes2image( brotli.decompress(drawing).decode("utf-8"), size=size, lw=line_width, linetype=linetype)
        else: 
            im = strokes2image( drawing, size=size, lw=line_width, linetype=linetype)
        ims.append( im )
    return ims

    #compressing and decompressing images
def convert_strokes2webp(drawings, size, line_width):
    "convert strokes to images"
    ims = []
    for drawing in drawings :            
        im = strokes2image( drawing, size=size, lw=line_width)
        with BytesIO() as byteIO:
            im = Image.fromarray(im).save(byteIO, format="webp", lossless=True ) #, quality=100)
            ims.append( byteIO.getvalue() )
    return ims

def decode_images(imgs:list):
    ims_de = []    
    for im in imgs :
        with BytesIO( im ) as byteIO:
            ims_de.append( np.asarray(Image.open(byteIO).convert("L")) )            
    return ims_de

def loadall(pickle_file):
    with open(pickle_file, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def readallImages(pickle_file ):
    "return a list of images compressed and save by merge_stroke_files"
    imgs = []
    for imgs_section in loadall(pickle_file):
        imgs.extend( imgs_section )
        gc.collect()    
    return imgs

# Numericalization of labels
# We try to get then same validation ratio for each label
def split_train_valid_in_place( df, ix_train_valid_col:int, valid_ratio:float):
    # increase valid_ratio is a bit high due to truncation that happens when doing randomization in groups
    valid_ratio = 1.06*valid_ratio
 
    k=0
    for n,g in df.groupby("word_code"):
        if len(g) > 1: 
            ix_valid = np.random.randint(0, len(g), size=int( len(g)*valid_ratio+.5) )
            df.iloc[g.index[ix_valid], ix_train_valid_col] = False 
    gc.collect()
    valid_ratio_measured = (1- sum(df["train_valid"]) / len(df) )
    print(f"Specificied validation ratio:{valid_ratio} measured ration:{valid_ratio_measured}" )

    