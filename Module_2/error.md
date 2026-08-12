## Task 1

error                                     Traceback (most recent call last)
Cell In[4], line 8
      5 img_bgr = cv2.imread(".\raw_captures\self_image_dim.jpg")
      7 # Convert color spaces
----> 8 img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
      9 img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
     10 img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

error: OpenCV(5.0.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'