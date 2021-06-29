# EasyOCR-vs-Tesseract-for-Vehicle-Plate-Recognition-System
This was a part of my final year project. The program displays accuracy of OCR on car images using Tesseract and EasyOCR.
The program was executed by giving Digital images of Car Number Plate acquired from Kaggle Dataset and Google Images. A folder contating some images is present in this repository.

## PROBLEM STATEMENT:

To comapre the accuracy of OCR on Vehicle Number Plates. 

OCR using Tesseract and EasyOCR Engines were performed. For comparing Accuracy we set the file name of images as the characters on the number plate. 
Tesseract offered better accuracy than EasyOCR.

## POST-PROCESSING:

Usually, number plates are not simple. Special Characters like ‘-’, ‘:’ between the characters corrupt the output and decrease accuracy. We also noticed that extra characters like ‘|’, ‘_’ ‘©’ etc. were detected due to shadows. These special characters were often present at the beginning or at the end of the string. To omit these characters, we put a check that the detected string should always start or end with an alphanumerical character, else the extra character should be dropped. This helped us improve the accuracy significantly. 

Besides, as the quality of pictures were not uniform, using filters with the same intensity values did not work for all images. 

## OBSERVATIONS

•	For all 200 images, the error is 16% from Tesseract and EasyOCR.
•	It is better to use EasyOCR with a GPU. Tesseract is faster than EasyOCR on CPU.
•	Removing special characters from the recognized image in post-processing significantly improved the accuracy. 
•	The total elapsed time of recognition is 1689.75 seconds. 
•	The average time of recognition of each image is 8.45 seconds on the CPU.
•	The plate status, environmental conditions and the hardware used to catch pictures are deterministic important factors for the proper functioning program.
•	A good image pre-processing almost guarantees successful recognition.
•	During Real-time execution, proper lighting and an accurate camera angle are important for proper capturing and recognition.
•	OCR is most effective when characters are in a contrasting background. 


## Future Scope of Work in the problem statement:
•	Compiling a better dataset for testing
•	Calculation of %Error for each function.
•	Use of GPU to check OCR performance.
•	Accuracy comparison of OCR engines in real-time. 
•	Images needs to be captured from public places for a better dataset.
•	Adaptive filtering needs to be tested.
•	End-to-end implementation of the project using hardware with a low-resolution camera in a parking lot.
