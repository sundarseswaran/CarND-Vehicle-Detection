import numpy as np
import math

def findHistPeaks(image, yTop, yBottom, Xleft, XRight):
    """ Finds a histogram with in the image to extract lanes

    """
    # find the histogram with in the image
    histogram = np.sum(image[yTop:yBottom,:], axis=0)

    # get max
    if len(histogram[int(Xleft):int(XRight)])>0:
        return np.argmax(histogram[int(Xleft):int(XRight)]) + Xleft
    else:
        return (Xleft + XRight) / 2

def doValidateLanes(lane, curverad, fitx, fit):
    """ Validate the detected lanes and set radius of curvature

    """

    if lane.detected:
        # If lane is detected, set Line parameters and rad.of.curvature
        if abs(curverad / lane.radius_of_curvature - 1) < .6:
            lane.detected = True
            lane.current_fit = fit
            lane.allx = fitx
            lane.bestx = np.mean(fitx)
            lane.radius_of_curvature = curverad
            lane.current_fit = fit
        else:
            lane.detected = False
            fitx = lane.allx
    else:
        # lane was not detected & curvature is defined
        if lane.radius_of_curvature:
            if abs(curverad / lane.radius_of_curvature - 1) < 1:
                lane.detected = True
                lane.current_fit = fit
                lane.allx = fitx
                lane.bestx = np.mean(fitx)
                lane.radius_of_curvature = curverad
                lane.current_fit = fit
            else:
                lane.detected = False
                fitx = lane.allx
        # no-lane, curvature was defined
        else:
            lane.detected = True
            lane.current_fit = fit
            lane.allx = fitx
            lane.bestx = np.mean(fitx)
            lane.radius_of_curvature = curverad
    return fitx


def doValidateDirection(right, right1, right2):
    """ Validate the identified vehicle direction

    """

    # get direction from the current values
    if abs((right-right1) / (right1-right2) - 1) < .2:
        return right
    else:
        # calculate based on the previous values
        return right1 + (right1 - right2)

def findLanes(n, image, x_window, lanes, leftLaneX, leftLaneY, rightLaneX, rightLaneY, window_ind, left_lane, right_lane):
    """ This function finds points/coordinates for left lane and right lane

        This fuction uses the binarized warped image to detect the lanes
    """

    # making assumptions for
    # left, right & center points for the warped image
    left, right = (300, 1100)
    center = 700
    center_pre = center
    direction = 0
    index1 = np.zeros((n+1,2))
    index1[0] = [300, 1100]
    index1[1] = [300, 1100]

    for i in range(n-1):

        # window range.
        yTop = 720-720/n*(i+1)
        yBottom = 720-720/n*i

        # left and right lanes are detected from the previous image
        if (left_lane.detected==False) and (right_lane.detected==False):
            # find historgram
            left  = findHistPeaks(image, yTop, yBottom, index1[i+1,0]-200, index1[i+1,0]+200)
            right = findHistPeaks(image, yTop, yBottom, index1[i+1,1]-200, index1[i+1,1]+200)

            # set direction
            left  = doValidateDirection(left, index1[i+1,0], index1[i,0])
            right = doValidateDirection(right, index1[i+1,1], index1[i,1])

            # set center
            center_pre = center
            center = (left + right)/2
            direction = center - center_pre
        else:
            # both the lanes are detected in the previous image
            left  = left_lane.windows[window_ind, i]
            right = right_lane.windows[window_ind, i]

        # ensure the distance between left and right lanes are wide enough
        if abs(left-right) > 600:
            # Append coordinates to the left lane arrays
            left_lane_array = lanes[(lanes[:,1]>=left-x_window) & (lanes[:,1]<left+x_window) &
                                 (lanes[:,0]<=yBottom) & (lanes[:,0]>=yTop)]
            leftLaneX += left_lane_array[:,1].flatten().tolist()
            leftLaneY += left_lane_array[:,0].flatten().tolist()

            if not math.isnan(np.mean(left_lane_array[:,1])):
                left_lane.windows[window_ind, i] = np.mean(left_lane_array[:,1])
                index1[i+2,0] = np.mean(left_lane_array[:,1])
            else:
                index1[i+2,0] = index1[i+1,0] + direction
                left_lane.windows[window_ind, i] = index1[i+2,0]

            # Append coordinates to the right lane
            right_lane_array = lanes[(lanes[:,1]>=right-x_window) & (lanes[:,1]<right+x_window) &
                                  (lanes[:,0]<yBottom) & (lanes[:,0]>=yTop)]
            rightLaneX += right_lane_array[:,1].flatten().tolist()
            rightLaneY += right_lane_array[:,0].flatten().tolist()
            if not math.isnan(np.mean(right_lane_array[:,1])):
                right_lane.windows[window_ind, i] = np.mean(right_lane_array[:,1])
                index1[i+2,1] = np.mean(right_lane_array[:,1])
            else:
                index1[i+2,1] = index1[i+1,1] + direction
                right_lane.windows[window_ind, i] = index1[i+2,1]

    return leftLaneX, leftLaneY, rightLaneX, rightLaneY