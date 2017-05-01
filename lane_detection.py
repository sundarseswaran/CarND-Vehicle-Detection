import cv2
import numpy as np
from lane_finder import *

def radiusOfCurvature(yValues, fitx):
    """ Calculate radius of curvature
    
    """

    ## define pixel convertions to meter unit
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700

    # calculate radius of curvature
    y_eval = np.max(yValues)
    fit_cr = np.polyfit(yValues*ym_per_pix, fitx*xm_per_pix, 2)
    radius = ((1 + (2*fit_cr[0]*y_eval + fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*fit_cr[0])
    return radius

def getVehiclePosition(pts, image_shape=(640, 1280)):
    """ Calculate the position of the car from the center of the lane 
    
    """

    # assume the car is in the center
    position = image_shape[1]/2
    leftArr = pts[(pts[:,1] < position) & (pts[:,0] > 600)][:,1]
    rightArr = pts[(pts[:,1] > position) & (pts[:,0] > 600)][:,1]
    
    # calculate deviation
    if(len(leftArr) > 0 and len(rightArr) > 0):
        left  = np.min(leftArr)
        right = np.max(rightArr)  
        center = (left + right)/2
    else:
        center = 0
    
    # pixel-meter conversion
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension    
    
    return (position - center)*xm_per_pix


def laneFitter(image, left_lane, right_lane):
    """ Fit lanes based on the window searching
    
    Find lanes with three different masks on the bottom portion of the image 
    and fit a polynomial based on the search results
    
    """

    # we may need to calculate X-values based on Y.
    # pick y-values for the entire height of the image
    yValues = np.linspace(0, 100, num=101)*7.2
    
    # coordinates for left & right lanes
    lanes = np.argwhere(image)
    lefLnX = []
    leftLnY = []
    rightLnX = []
    rightLnY = []

    # Curving left or right - -1: left 1: right
    curve = 0

    # Set left and right as None
    left = None
    right = None

    # Find lanes from three repeated procedures with different window values
    lefLnX, leftLnY, rightLnX, rightLnY = findLanes(4, image, 25, lanes, lefLnX, leftLnY, rightLnX, rightLnY, 0, left_lane, right_lane)
    lefLnX, leftLnY, rightLnX, rightLnY = findLanes(6, image, 50, lanes, lefLnX, leftLnY, rightLnX, rightLnY, 1, left_lane, right_lane)
    lefLnX, leftLnY, rightLnX, rightLnY = findLanes(8, image, 75, lanes, lefLnX, leftLnY, rightLnX, rightLnY, 2, left_lane, right_lane)

    # computer coefficients for the polynomial
    leftFit = np.polyfit(leftLnY, lefLnX, 2)
    leftFitX = leftFit[0]*yValues**2 + leftFit[1]*yValues + leftFit[2]
    rightFit = np.polyfit(rightLnY, rightLnX, 2)
    rightFitX = rightFit[0]*yValues**2 + rightFit[1]*yValues + rightFit[2]

    # get radius of curvature
    left_curverad  = radiusOfCurvature(yValues, leftFitX)
    right_curverad = radiusOfCurvature(yValues, rightFitX)
    
    # validate lanes' fit
    leftFitX  = doValidateLanes(left_lane, left_curverad, leftFitX, leftFit)
    rightFitX = doValidateLanes(right_lane, right_curverad, rightFitX, rightFit)
    
    return yValues, leftFitX, rightFitX, lefLnX, leftLnY, rightLnX, rightLnY, left_curverad




def drawOverlay(image, warped, yValues, leftFitX, rightFitX, 
              lefLnX, leftLnY, rightLnX, rightLnY, inv_M, curvature):

    """ Draw an overlay on the predicted lanes 
    
    """
    
    # overlay template
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # stack left lane & right lane together
    pt_left = np.array([np.transpose(np.vstack([leftFitX, yValues]))])
    pt_right = np.array([np.flipud(np.transpose(np.vstack([rightFitX, yValues])))])
    pts = np.hstack((pt_left, pt_right))
    
    # draw lanes onto the template - in green color
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 200, 0))
    newwarp = cv2.warpPerspective(color_warp, inv_M, (image.shape[1], image.shape[0])) 
    
    # apply overlay
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    # add caption
    text = "Radius of Curvature: {:.2f} km".format(curvature/1000)
    cv2.putText(result,text,(100,100), 1, 1,(255,255,255),2)
    
    # find the poisition of the car with respect to lane center
    pts = np.argwhere(newwarp[:,:,1])
    position = getVehiclePosition(pts)
    
    if position < 0:
        text = "Vehicle is {:.2f} m left of lane center".format(-position)
    else:
        text = "Vehicle is {:.2f} m right of lane center".format(position)
    
    cv2.putText(result,text,(100,150), 1, 1,(255,255,255),2)

    return result

