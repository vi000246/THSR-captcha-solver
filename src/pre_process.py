# -*- coding: utf-8 -*-
"""pre_process module

This module include utils for pre_process.

Reference
    1. https://youtu.be/6HGbKdB4kVY
    2. https://youtu.be/4DHcOPSfC4c
"""
import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import binarize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

def pre_process(path, remove_curve=True):
    """Denoise and remove curve(option)

    Args:
        path (str): image pathh.
        remove_curve (bool): remove curve or not.

    Returns:
        Image: the image after preprocess.
    """
    img = cv2.imread(path)
    h, w, _ = img.shape
    dst = cv2.fastNlMeansDenoisingColored(img, None, 30, 30 , 7 , 21)
    ret,thresh = cv2.threshold(dst,127,255,cv2.THRESH_BINARY_INV)
    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    if not remove_curve:
        return Image.fromarray(imgarr) 

    # remove curve
    imgarr[:,5:w-5] = 0
    imagedata = np.where(imgarr == 255)

    X = np.array([imagedata[1]])
    Y = 47 - imagedata[0]

    poly_reg= PolynomialFeatures(degree = 2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)

    X2 = np.array([[i for i in range(0,w)]])
    X2_ = poly_reg.fit_transform(X2.T)

    newimg =  cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    for ele in np.column_stack([regr.predict(X2_).round(0),X2[0],] ):
        pos = 47-int(ele[0])
        newimg[pos-2:pos+4,int(ele[1])] = 255 - newimg[pos-2:pos+4,int(ele[1])]
    return Image.fromarray(newimg) 

def main():
    """!!! useless !!!
    """
    img = cv2.imread('../captcha/01998.png')
    h, w, _ = img.shape
    dst = cv2.fastNlMeansDenoisingColored(img, None, 30, 30 , 7 , 21)
    ret,thresh = cv2.threshold(dst,127,255,cv2.THRESH_BINARY_INV)
    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    imgarr[:,5:w-5] = 0

    cv2.imwrite('output.png', thresh)
    imagedata = np.where(imgarr == 255)

    X = np.array([imagedata[1]])
    Y = 47 - imagedata[0]

    poly_reg= PolynomialFeatures(degree = 2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)

    X2 = np.array([[i for i in range(0,w)]])
    X2_ = poly_reg.fit_transform(X2.T)


    newimg =  cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    for ele in np.column_stack([regr.predict(X2_).round(0),X2[0],] ):
        pos = 47-int(ele[0])
        # if newimg[pos-4:pos+4,int(ele[1])] == 255:
        # newimg[pos-3:pos+4,int(ele[1])] = 0 
        newimg[pos-2:pos+4,int(ele[1])] = 255 - newimg[pos-2:pos+4,int(ele[1])]
    
    cv2.imwrite('output50.png', newimg)
    dst = cv2.fastNlMeansDenoising(newimg, None, 30 , 7 , 21)
    cv2.imwrite('outputdst.png', dst)
    
if __name__ == "__main__":
    main()