import numpy as np
import cv2 as cv

"""

Defining the used functions to make the code neet and readable.
Task 1 is to print the union of two images that have intersection for sure.
The Task of printing the intersection can be done in the same code.

"""


## Function to detect the difference between two frames.
def diff(prev, frame):

    # prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # | Compute the Mean Squared Error (MSE) |
    mse = np.mean((prev - frame) ** 2)
    print(mse) ## for debugging purpose only
    return mse

## Function to find out the intersection of two images and return the common part of both the left and right frame as well as the indices of the seperator.
def search(r, l, stype: str = "left", dif: float = 0.1): ### If stype is "left", the function performs search in the left image/array.
  length = len(r)
  a = []
  index = None
  if stype == 'left':
    for i in range(length):
      if diff(l[:,i,:],r[:,0,:])<=dif:
        a = l[:,i:,:]
        index = i
        break
  elif stype == 'right':
    for i in range(length):
      i = length -i -1
      if diff(r[:,i,:],l[:,-1,:])<=dif:
        a = r[:,0:i+1,:]
        index = i
        break
  # print(a,"A") ### debuging only
  return [a,index]

## Function to seperate find and verify the union and intersection of two frames and return them.
## dif parameter needs to be tuned for perfect concatenation of images horizontally.
def union(imgl, imgr, dif:float = 0.1):
  l = len(imgl)
  r = len(imgr)
  if l != r: ### | Condition to check if the images are of same size or not |
    print("Error!!!, Images are of different shape")
    return None
  # union = imgl.copy()
  left_comn, l_index = search(imgr, imgl, stype = 'left', dif=dif)
  left_comn = np.array(left_comn)
  print(l_index,"left_comn" ) ## debuging only
  right_comn, r_index = search(imgr, imgl, stype = 'right', dif=dif)
  right_comn = np.array(right_comn)
  print(r_index,"right_comn" ) ## debuging only
  if len(left_comn) == len(right_comn):
    union = np.concatenate((imgl, imgr[:,r_index+1:,:]), axis=1)
  else:
    print("Error!!!, Images are of different shape",len(left_comn), len(right_comn))
    return None
  return [union,left_comn, right_comn]



"""
Using the above functions to seperate frames.
"""
orig_img = cv.imread("glory.jpg")
imgl = cv.imread("left.png")
# print(imgl.shape)
imgr = cv.imread("right.png")
# print(imgr.shape)


cv.imshow("Original image",orig_img)
cv.imshow("Left Frame",imgl)
cv.imshow("Right Frame",imgr)

results = union(imgl,imgr)
# print(type(results),1)
cv.imshow("Union of the Images",results[0])
# print(type(results[0]),2)
cv.imshow("Intersection in left part",np.array(results[1]))
# print(type(results[1]),3)
cv.imshow("Intersection in Right part",np.array(results[2]))
# print(type(results[2]),4)

cv.waitKey(0)
cv. destroyAllWindows()
