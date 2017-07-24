# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[image1]: ./writeup_images/calibration_results.png "Undistorted"
[image2]: ./writeup_images/undistortion_test.png "Road Transformed"
[image3]: ./writeup_images/combined_thresh.png "Binary Example"
[image4]: ./writeup_images/perspective_test.png "Warp Example"
[lines]:  ./output_images/persp_lines.jpg "Warp method"
[image5]: ./output_images/window_fit.jpg "Fit Visual"
[image6]: ./writeup_images/results.png "Output"
[video1]: ./project_video.mp4 "Video"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. You can use template writeup for this project as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained the IPython notebook located in "./camera_cal/Camera Calibration.ipynb", which was created for this purpose only.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

The calibration matrix and the distortion coefficients were then saved to the wide_dist_pickle.p file so I wouldn't have to repeat the calibration process later.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

On the first cell of the Project.ipynb, I loaded the calibration matrix and the distortion coefficients that were obtained on the camera calibration and applied the distortion correction fuction `cv2.undistort()`, this time to a road image.

It is possible to notice the camera distortion on the sides of the image, on the tree and on the white car.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image using the functions defined in the code cells 4 through 7.  I played a bit with the thresholds to see how I could isolate better the lane line pixels. Here is a screeshots of one of these attempts.

![alt text][image3]

I ended up using a combination of saturation HLS channel, value HSV channel, sobel x and sobel y thresholds.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the 2nd code cell of the IPython notebook Project.ipynb.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.

The source points were chosen by drawing lines on the image and trying to make them match the lane lines like shown on the image below: 

![alt text][lines]

At the end the four source points chosen were:

```python
src = [[  590.   455.]  # top left
       [  270.   680.]  # bottom left 
       [ 1054.   680.]  # bottom right
       [  698.   455.]] # top right
```

For the destination points I found that a good approximation was to plot one of the lines at 1/4 of the width and the other at 3/4 all the way from zero to its full height:

```python
dst = np.float32([
    (img.shape[1]/4, 0), # top left
    (img.shape[1]/4, img.shape[0]), # bottom left
    (img.shape[1]*3/4, img.shape[0]), # bottom right
    (img.shape[1]*3/4, 0)]) # top right
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 455      | 320, 0        | 
| 270, 680      | 320, 720      |
| 1054, 680     | 960, 720      |
| 698, 455      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After running the perspective transform and obtaining the warped image, I implemented a sliding window search which consists of a technique that divides the image in windows and for each window it averaged the x coordinates of the pixels to find the line centroids. After the search scanned all the warped image, it combined the centroids to draw the lane lines. The result of the sliding windows search was as follows

![alt text][image5]

I then fit a 2nd order polynomial to these lane lines using `numpy.polyfit()`

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used the formula provided in the lesson to calculate the radius of curvature. This line of code is where the magic happened.

```python
    ym_per_pix = 10/720
    xm_per_pix = 4/384
    curve_fit_cr = np.polyfit(np.array(res_yvals, np.float32)*ym_per_pix, np.array(leftx,np.float32)*xm_per_pix, 2)
    curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5)/np.absolute(2*curve_fit_cr[0])
```
The first two lines consisted of ratios to convert pixels to meters in real world dimensions. The othe two lines consisted of the actual radius of curvature calculation which was done in two steps: another polynomial fit and the radius of curvature equation for a given curve.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the code cells 12 and 15.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./final_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline still needs a lot of manual parameter tuning and once tuned these parameters keep the same value forever.

This makes it not so robust and flexible to different pavement colors or big changes in light, even though it does much bettter than the first projects pipeline.

If there was a way to make the pipeline learn to adapt to different environments I guess that would make it more robust. That can be achieved with the use of other sensors, not just cameras.
