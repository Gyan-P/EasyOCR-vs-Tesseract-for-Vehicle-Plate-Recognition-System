import cv2
import pytesseract #ImgToOCR
from matplotlib import pyplot as plt
import argparse
from PIL import Image
import os
import imutils
import glob
import numpy as np
import easyocr #ImgToOCR 
import time 

def main():
    # specify path to the license plate images folder as shown below
    path_for_license_plates = os.getcwd() + "/license_plates/*.jpg"
    

    for path_to_license_plate in glob.glob(path_for_license_plates, recursive = True):
     
        license_plate_file=os.path.basename(path_to_license_plate)
        license_plate, _ = os.path.splitext(license_plate_file)
        print(license_plate)
        '''
        Here we append the actual license plate to a list
        '''
        list_license_plates.append(license_plate)

        '''
        Read each license plate image file using openCV
        '''
        raw_img = cv2.imread(path_to_license_plate)
        #plt.imshow(raw_img)
        img = filtering(raw_img)
        
        '''
        We then pass each license plate image file
        to the Tesseract OCR engine using the Python library
        wrapper for it. We get back predicted_result for
        license plate. We append the predicted_result in a
        list and compare it with the original the license plate
        '''
        func_tesseract(img)
        func_EasyOCR(img)
        
        

#This function pre-processses the image by applying various filters and morphological operations
def filtering(img):
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    
    bfilter = cv2.bilateralFilter(gray, 60, 50, 20)
    #plt.imshow(cv2.cvtColor(bfilter,cv2.COLOR_BGR2RGB))
    
    edged = cv2.Canny(bfilter, 70, 50) #Edge detection
    #plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        global approx
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
        #print(location)
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0,255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    (x,y) = np.where(mask==255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
        
        
    #plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    #p=input("Press 1 to continue")
    #reader = easyocr.Reader(['en'])
    #result = reader.readtext(cropped_image)
    
    
    return cropped_image



#Function to perform OCR using Teserract
def func_tesseract(cropimg):
    predicted_result = pytesseract.image_to_string(cropimg, lang ='eng', config='--psm 6')
    print(predicted_result)
    filter_predicted_result = "".join(predicted_result.split()).replace(" ", "").replace("-", "")
    first_char=filter_predicted_result[0]
    last_char=filter_predicted_result[-1]
    if(not((ord(first_char)>=65 and ord(first_char)<=90) or (ord(first_char)>=48 and ord(first_char)<=57))):
        filter_predicted_result= filter_predicted_result[1:]
    if(not((ord(last_char)>=65 and ord(last_char)<=90) or (ord(last_char)>=48 and ord(last_char)<=57))):
        filter_predicted_result= filter_predicted_result.rstrip(filter_predicted_result[-1])
    print(filter_predicted_result)
    predicted_license_plates_tess.append(filter_predicted_result)
    return

#Function to perform OCR using EasyOCR
def func_EasyOCR(cropimg):
    reader = easyocr.Reader(['en'])
    predicted_result = reader.readtext(cropimg, detail=0, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    text=predicted_result[0]
    print(text)
    predicted_license_plates_eo.append(text)
    return

#Function to print the accuracy obtained from tesseract 
def calculate_predicted_accuracy(actual_list, predicted_list):
        for actual_plate, predict_plate in zip(actual_list, predicted_list):
            accuracy = "0 %"
            num_matches = 0
            if actual_plate == predict_plate:
                accuracy = "100 %"
            else:
                if len(actual_plate) == len(predict_plate):
                    for a, p in zip(actual_plate, predict_plate):
                        if a == p:
                            num_matches += 1
                    accuracy = str(round((num_matches / len(actual_plate)), 2) * 100)
                    accuracy += "%"
            print("	 ", actual_plate, "\t\t\t", predict_plate, "\t\t ", accuracy)




#Function to print the accuracy obtained from EasyOCR


def calculate_predicted_accuracy_EasyOCR(actual_list, predicted_list):
        for actual_plate, predict_plate in zip(actual_list, predicted_list):
            accuracy = "0 %"
            num_matches = 0
            if actual_plate == predict_plate:
                accuracy = "100 %"
            else:
                if len(actual_plate) == len(predict_plate):
                    for a, p in zip(actual_plate, predict_plate):
                        if a == p:
                            num_matches += 1
                    accuracy = str(round((num_matches / len(actual_plate)), 2) * 100)
                    accuracy += "%"
            print("	 ", actual_plate, "\t\t\t", predict_plate, "\t\t ", accuracy)
            
if __name__ == "__main__":
    list_license_plates = []
    predicted_license_plates_tess = []
    predicted_license_plates_eo = []
    t1=time.time()
    main()
    t2=time.time()-t1
    print(t2)
    print("Tesseract Accuracy")
    print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy")
    print("--------------------", "\t", "-----------------------", "\t", "--------")
    calculate_predicted_accuracy(list_license_plates, predicted_license_plates_tess)
    print("EasyOCR Accuracy")
    print("Actual License Plate", "\t", "Predicted License Plate", "\t", "Accuracy")
    print("--------------------", "\t", "-----------------------", "\t", "--------")
    calculate_predicted_accuracy(list_license_plates, predicted_license_plates_eo)
