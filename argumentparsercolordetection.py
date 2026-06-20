# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 19:01:52 2026

@author: user
"""

import argparse

import pandas as pd

import numpy as np

import cv2


ap = argparse.ArgumentParser()

ap.add_argument('-i','--image',required=True, 
                help='Image File Name')

args = vars(ap.parse_args())

img_path = args['image']

#image read
img  = cv2.imread(img_path)

clicked = False

r = g = b = xpos = ypos = 0

index = ['color','color_name','hexid','R','G','B']

csv = pd.read_csv('colors.csv', names = index, header = None)

# Calculate the min distance between all the colors and get the matching color

def getcolorname(R,G,B):
    min = 1000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i,'R'])) + abs(G - int(csv.loc[i,'G'])) 
        + abs(B - int(csv.loc[i,'B']))
        
        
        if (d<min):
            min = d
            cname = csv.loc[i,'color_name']
    return cname

def draw(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        
        global r,g,b,xpos,ypos,clicked
        
        clicked = True
        
        xpos = x
        ypos = y 
        b,g,r = img[y,x]
        
        r = int(r)
        g = int(g)
        b = int(b)
        
        
cv2.namedWindow('image')

cv2.setMouseCallback('image', draw)


while True:
    cv2.imshow('Image',img)
    
    if(clicked == True):
        cv2.rectangle(img,(20,20),(750,60),(b,g,r),-1)
        text = getcolorname(r,g,b)+'R='+str(r)+'G='+str(g)+'B='+str(b)
        cv2.putText(img, text, (60,60), 2, 0.8, (255,255,255),2, cv2.LINE_AA)
        
        if(r+g+b>=600):
            cv2.putText(img, text, (60,60), 2, 0.8, (0,0,0),2, cv2.LINE_AA)
        clicked = False
    
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()


        
        
        
        






