import numpy as np
import cv2 as cv
import glob



################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,6)
#frameSize = (1024,1280)
frameSize = (1280,1024)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


imagesLeft = glob.glob('images/left/*.jpg')
imagesRight = glob.glob('images/right/*.jpg')

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1000)


cv.destroyAllWindows()




############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
print("Left Camera Matrix:\n", cameraMatrixL)
print("Left Distortion Coefficients:\n", distL)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
print("Right Camera Matrix:\n", cameraMatrixR)
print("Right Distortion Coefficients:\n", distR)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

k1_L, k2_L = distL[0, 0], distL[0, 1]
#print("Left Camera k1:", k1_L)
#print("Left Camera k2:", k2_L)
print("Left Camera Matrix:\n", cameraMatrixL)

# Extract and print k1 and k2 for the right camera
k1_R, k2_R = distR[0, 0], distR[0, 1]
#print("Right Camera k1:", k1_R)
#print("Right Camera k2:", k2_R)
print("Right Camera Matrix:\n", cameraMatrixR)



print("Saving parameters!")
########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)

#print(newCameraMatrixL)
#print(newCameraMatrixR)

########## Stereo Rectification #################################################

rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Q is", Q)
print("Rotation Matrix:\n", rot)
print("Translation Matrix:\n", trans)


print("rectification Left", rectL)
print("rectification Right", rectR)
print("Projection left", projMatrixL)
print("Projection Right", projMatrixR)

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()

imgL = cv.imread('C:/Users/Nazanin Padkan/Desktop/Gupho code/KE0200020142/1663615376199.jpg')
imgR = cv.imread('C:/Users/Nazanin Padkan/Desktop/Gupho code/KE0200020137/1663615376199.jpg')

rectifiedImgL = cv.remap(imgL, stereoMapL[0], stereoMapL[1], cv.INTER_LINEAR)
rectifiedImgR = cv.remap(imgR, stereoMapR[0], stereoMapR[1], cv.INTER_LINEAR)

cv.imwrite('rectified_left_image.jpg', rectifiedImgL)
cv.imwrite('rectified_right_image.jpg', rectifiedImgR)

cv.imshow('Rectified Left Image', rectifiedImgL)
cv.imshow('Rectified Right Image', rectifiedImgR)
cv.waitKey(0)
cv.destroyAllWindows()
