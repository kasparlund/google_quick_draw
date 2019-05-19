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
import pickle
import gc
from core_ai.lib.data import *

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

def drawing2image_nomalized(side_size:int, line_width:int, linetype:int, compressed_drawings:bool,
                     image_mean:float=0, image_sd:float=1):
    image_mean, image_sd = image_mean*255, image_sd * 255
    def _inner(drawing): 
        im  = strokes2image( brotli.decompress(drawing).decode("utf-8"), size=side_size, lw=line_width, linetype=linetype)
        im = (np.asarray( im, dtype=np.float32)-image_mean ) / image_sd 
        return im.reshape(-1,side_size,side_size)
    return _inner

def drawing2image(drawing, size, line_width, linetype, compressed_drawings):
    if compressed_drawings:
        im = strokes2image( brotli.decompress(drawing).decode("utf-8"), size=size, lw=line_width, linetype=linetype)
    else: 
        im = strokes2image( drawing, size=size, lw=line_width, linetype=linetype)
    return im

def drawings2images(drawings, size, line_width, linetype, compressed_drawings):
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


class NormalizedDataset(Dataset):
    def __init__(self, x, y, qd2norm_image): 
        self.x,self.y, self.qd2norm_image = x,y,qd2norm_image
    def __getitem__(self, i): 
        return  self.qd2norm_image(self.x[i]), int(self.y[i])

class GDDataset(Dataset):
    def __getitem__(self, i): 
        return  self.x[i], int(self.y[i])


def readDatasets(train_ds_file:Path, valid_ds_file:Path):
    with train_ds_file.open("rb") as file:
        train_ds = pickle.load(file)
    with valid_ds_file.open("rb") as file:
        valid_ds = pickle.load(file)
    gc.collect()
    return train_ds, valid_ds


def calc_x_mean( ds:Dataset, qd2image ): 
    x_mean   = 0
    for x,y in progress_bar(ds):
        x_mean += qd2image(x).flatten().mean()
    return x_mean / len(train_ds)

def calc_x_sd(ds:Dataset, x_mean:float, gd2image):
    x_sd     = 0
    for x,y in progress_bar(ds):
        x_sd += np.power( gd2image(x).flatten() - x_mean, 2).mean()
    return math.sqrt( x_sd / len(train_ds) )