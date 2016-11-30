
import xmltodict
import cv2
import os

path = r"C:\Users\westwell\Desktop\ST\ST\data\Annotations\00000.xml"
with open(path) as fd:
    doc = xmltodict.parse(fd.read())

mapDict = {'one':1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9, "zero":0}

for i in range(len(doc['annotation']['object'])):
	print mapDict[ doc['annotation']['object'][i]['name'] ]
