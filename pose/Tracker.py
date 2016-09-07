#! /usr/bin/env python

from __future__ import division, print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os.path

cv2.setNumThreads(-1)

class Tracker:
	
	"""Initializes a Tracker. Opens the source to check if it is working and then closes it.

	Parameters
	----------
	video_source : int or str
		The video source for the cv2.VideoCapture class. 
		This can be an int for a webcam or a filepath string to a video file.

	Returns
	-------
	Tracker : class
		the Tracker class 

	"""
	def __init__(self, video_source):
		

		assert type(video_source) in [int, str],  "video_source must be type int or str"
		
		self.source = video_source
		self.cap = cv2.VideoCapture(self.source)
		cv2.waitKey(100)

		assert self.cap.isOpened(), "Video source failed to open. \nCheck that your camera is connected or that the video file exists."
		ret, frame = self.cap.read()
		assert ret == True, "Video source opened but failed to read any images. \nCheck that the camera is working or the video is a valid file"
		

		self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

		self.pt1 = (0,0)
		self.pt2 = (self.width-1,self.height-1)
		
		if type(self.source) == str:
			self.fps = self.cap.get(cv2.CAP_PROP_FPS)
		else:
			self.fps = 29.9

		self.delay = 1./self.fps

		self.area_min = 10
		self.area_max = 1000

		self.bg_history = 1000
		self.bg_threshold = 200

		self.blur_size = 1
		self.blur_sigma = 0

		self.erosion_size = 1
		self.dilation_size = 1

		self.savefile = None

		self.cap.release()
		cv2.waitKey(100)

	def _crop(self, image, pt1, pt2):
		"""Returns an imaged cropped to a region of interest.

		Parameters
		----------
		image : array
			array image data
		pt1 : tuple of int
			top left point of the cropped area as (x,y)
		pt2 : boolean
			bottom right point of the cropped area (x,y)

		Returns
		-------
		cropped : array
			the cropped image 

		"""

		cropped = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]

		return cropped

	def show_video(self, resize = 1, video_output = None):

		"""Shows video source. Press `esc` to exit

			Parameters
			----------
			resize : int or float, default = 1
				resize factor for showing the video source. 1 = original size
			video_output : {str, None}, default = None
				Saves the stream to a file as .mp4, i.e. "/path/to/video.mp4".
				If None, no video is saved.
		"""

		assert type(resize) in [int, float], "resize must be int or float"
		if video_output != None:
			assert os.path.splitext(video_output)[1] in [".MP4",".mp4"], "Video output file must be .mp4"
		
		print("Showing video source. Press `esc` to exit.")

		if video_output != None:
			print("Saving video to " + os.path.basename(video_output) + "...")

		cap = cv2.VideoCapture(self.source)

		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*resize)
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*resize)

		if video_output != None:
			fourcc = cv2.VideoWriter_fourcc("M","P","4","V")
			out = cv2.VideoWriter(video_output, fourcc, self.fps, (width, height), True)

		video_title = "Video (press `esc` to exit)."
		if video_output != None:
			video_title = video_title + " RECORDING"

		cv2.startWindowThread()
		EXIT = False
		while cap.isOpened() and EXIT == False:
			ret,frame = cap.read()
			if ret:
				frame = cv2.resize(frame, (width,height), None) 
				
				if video_output != None:
					out.write(frame)
				cv2.imshow(video_title, frame)

			elif cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
				EXIT = True

			if cv2.waitKey(1) & 0xff == 27:
				EXIT = True

			if EXIT == True:
				break

		if video_output != None:
			out.release()

		cap.release()

		cv2.waitKey(1)
		cv2.destroyAllWindows()
		cv2.waitKey(1)

		for i in range(10):
			cv2.waitKey(1)



		print("Succesfully exited.")

	def show_still(self, frame_number = None, figsize = (10,10), still_output = None):

		"""Capture and show a still image from the video source.
			Parameters
			----------
			frame_number : int, default = None
				Frame number to show from the video 
			figsize : tuple of int, default = (10,10)
				Figure size for showing the image with matplotlib
			still_output : {str, None}, default = None
				Saves the image to a file. If None, no image is saved.
		"""

		assert type(still_output) in [type(None), str], "still_output must be type str or None"
		if still_output != None:
			assert os.path.splitext(still_output)[1] in [".jpg",".png",".jpeg",".tiff"], "Image file must be .jpg, .jpeg, .png, or .tiff"
		assert type(frame_number) in [type(None), int], "frame_number must be type int or None"
		if frame_number != None:
			assert type(self.source) in [str], "Cannot specify frame_number with camera source" 
		

		cap = cv2.VideoCapture(self.source)
		cv2.waitKey(100)

		if frame_number != None:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

		ret, still = cap.read()
		cap.release()
		plt.figure(figsize = (10,10))
		still = cv2.cvtColor(still, cv2.COLOR_BGR2RGB)
		plt.imshow(still)
		plt.title(str(still.shape[1]) +"x"+ str(still.shape[0]))
		plt.show()

		if still_output != None:
			cv2.imwrite(still_output, still)

	def set_roi(self, pt1, pt2, imshow = True, figsize = (20,6), frame_number = None):
		"""Set the region of interest for the tracker.

			Parameters
			----------
			pt1 : tuple of int
				top left point (x,y) of the region of interest
			pt2 : tuple of int
				bottom right point (x,y) of the region of interest
			imshow : bool, default = True
				Shows a image with the area of interest drawn over top
			figsize : tuple of int, default = (20,6)
				The size of the imshow image
			frame_number : int, default = None
				Frame number to show from the video 
		"""

		assert type(pt1) == tuple and type(pt2) == tuple, "pt1 and pt2 must be type tuple"
		assert len(pt1) == 2 and len(pt2) == 2, "pt1 and pt2 must be length 2"
		assert (type(pt1[0]) == int 
			and type(pt1[1]) == int
			and type(pt2[0]) == int
			and type(pt2[1]) == int), "pt1 and pt2 must contain only type int"
		assert type(frame_number) in [type(None), int], "frame_number must be type int or None"
		if frame_number != None:
			assert type(self.source) in [str], "Cannot specify frame_number with camera source"

		self.pt1 = pt1
		self.pt2 = pt2

		cap = cv2.VideoCapture(self.source)
		cv2.waitKey(100)

		if frame_number != None:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

		ret, still = cap.read()
		cap.release()

		still = cv2.cvtColor(still, cv2.COLOR_BGR2RGB)
		cropped = self._crop(still, pt1, pt2)

		self.height = cropped.shape[0]
		self.width = cropped.shape[1]

		if imshow == True:
			fig, (ax1,ax2) = plt.subplots(1, 2, figsize = figsize)
			roi = cv2.rectangle(still.copy(), pt1, pt2, (0,255,0), 3)
			ax1.imshow(roi)
			ax1.set_title("Region of Interest")

			ax2.imshow(cropped)
			ax2.set_title(str(self.width) +"x"+ str(self.height))
			plt.tight_layout()
			plt.show()

	def set_contour_area(self, area_min = 10, area_max = 1000):
		"""Set the contour area range for the tracker.

			Parameters
			----------
			area_min : int, default = 10
				Minimum area a contour must have to be tracked
			area_max : int, default = 1000
				Maximum area a contour can have to be tracked
		"""

		assert type(area_min) is int, "area_min must be type int"
		assert type(area_max) is int, "area_max must be type int"

		self.area_min = area_min
		self.area_max = area_max

	def set_background(self, history = 1000, threshold = 200):
		"""Set the background model parameters for the tracker.

			Parameters
			----------
			history : int, default = 1000
				Number of previous frames to use for building the background model.
				Higher values create a more conservative background model, but the model does not adapt quickly to changes in the scene.
				Lower values create a less conservative background model, but the model may adapt too quickly making the foreground part of the background model.
			threshold : int, default = 200
				Threshold value of the squared distance between foreground and background. 
				Higher values give more conservative segmentation.
				Lower values give less conservative segmentation but may introduce noise.
		"""

		assert type(history) is int, "history must be type int"
		assert type(threshold) is int, "threshold must be type int"

		self.bg_history = history
		self.bg_threshold = threshold

	def set_fps(self, fps = 29.9):
		"""Set the fps for the tracker.

			Parameters
			----------
			fps : int or float, default = 29.9
				The framerate for the tracker in Hz
		"""
		assert type(fps) in [int, float], "fps must be type int or float"

		self.fps = fps
		self.delay = 1./self.fps

	def set_blur(self, blur_size = 1, blur_sigma = 0):

		"""Set the Gaussian blur parameters for the tracker. Blur reduces noise in the frame.

			Parameters
			----------
			blur_size : int, default = 1
				Size of the Gaussian kernel to use for blur.
			blur_sigma : int, default = 0
				Variance of the Gaussian kernel to use for blur.
		"""

		assert type(blur_size) in [int], "blur_size must be type int"
		assert blur_size % 2 == 1, "blur_size must be an odd number (blur_size % 2 == 1)"

		assert type(blur_sigma) in [int], "blur_sigma must be type int"

		self.blur_size = blur_size
		self.blur_sigma = blur_sigma

	def set_morphology(self, erosion_size = 1, dilation_size = 1):

		"""Set the morphological transformations for the tracker.

			Parameters
			----------
			erosion_size : int, default = 1
				Size of the kernel to use for erosion. Erosion removes noise smaller than the kernel.
			dilation_size : int, default = 1
				Size of the kernel to use for dilation. Dilation expands pixels smaller than the kernel to the size of the kernel.
		"""

		assert type(erosion_size) in [int], "erosion_size must be type int"
		assert type(dilation_size) in [int], "dilation_size must be type int"

		self.erosion_size = erosion_size
		self.dilation_size = dilation_size


	def track(self, data_output = None, video_output = None, show_bg = True, show_mask = True, vtrail_len = 10):

		"""Starts the tracker with options to save the tracking data and record video to a file.

			Parameters
			----------
			data_output : str, default = None
				File path to the text file where the data will be stored as comma separated values.
				Must be file extension .txt or .csv
				If None, no data will be stored.
			video_output. : str, default = None
				File path to the video file where the output video will be stored as MPEG-4.
				Must be file extension .mp4 or .MP4
				If None, no video will be recorded.
			show_bg : bool, default = True
				Show the background model
			show_mask : bool, default = True
				Show the binarized foreground mask.
				This is produced by segmenting the current frame from the background model.
			vtrail_len : int, default = 10
				Length of the vanishing trail (number of points) in the tracking visualization
		"""

		assert type(data_output) in [str, type(None)], "data_output must be type str or None"
		if data_output != None:
			assert os.path.splitext(data_output)[1] in [".csv",".txt"], "Data file must be .csv, or .txt"

		assert type(video_output) in [str, type(None)], "video_output must be type str or None"
		if video_output != None:
			assert os.path.splitext(video_output)[1] in [".mp4",".MP4"], "Video output file must be .mp4"
		
		assert type(show_bg) in [bool], "show_bg must be type bool"
		assert type(show_mask) in [bool], "show_mask must be type bool"

		timestamp = dt.datetime.now()

		if data_output != None:
			self.savefile = open(data_output,"w")

			if type(self.source) == str:
				write_str = "position_msec" + "," + "frame_number" + "," + "x" + "," + "y" + "\n"
			elif type(self.source) == int:
				write_str = "date_time" + "," + "frame_number" + "," + "x" + "," + "y" + "\n"
			
			self.savefile.write(write_str)

		fwidth = int(np.sum([True, show_bg, show_mask]))

		cap = cv2.VideoCapture(self.source)

		total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

		ekernel = np.ones((self.erosion_size,self.erosion_size),np.uint8)
		dkernel = np.ones((self.dilation_size,self.dilation_size),np.uint8)

		fgbg = cv2.createBackgroundSubtractorKNN(history = self.bg_history, dist2Threshold = self.bg_threshold, detectShadows = False)

		if video_output != None:
			fourcc = cv2.VideoWriter_fourcc("A","V","C","1")
			out = cv2.VideoWriter(video_output, fourcc, self.fps, (self.width*fwidth,self.height), True)

		if type(self.source) == int:
			frame_number = 0

		vanishing_trail = np.ones((vtrail_len,2))*-1

		cv2.namedWindow("Tracker")
		cv2.startWindowThread()
		while cap.isOpened():

			mcx,mcy,o1 = -1,-1,-1
			ret, frame = cap.read()

			if type(self.source) == int:
				frame_number += 1
			if type(self.source) == str:
				frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

			if ret:
				frame = self._crop(frame, self.pt1, self.pt2)
				time =  dt.datetime.now()
				
				if self.blur_size != None and self.blur_sigma != None:
					blur = cv2.GaussianBlur(frame, (self.blur_size, self.blur_size), self.blur_sigma) 
				else:
					blur = frame

				fgmask = fgbg.apply(blur)

				if self.erosion_size != None:
					erosion = cv2.erode(fgmask, ekernel, iterations = 1) 
				else:
					erosion = fgmask
				if self.dilation_size != None:
					dilation = cv2.dilate(erosion, dkernel, iterations = 1)
				else:
					dilation = erosion 

				mask = dilation

				_, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # get contours

				if len(contours) > 0 and len(contours) <= 5: # if contours are found, but not too many...
						
					# find contour with min area and store it as best_cnt...

					max_area = 0 # set intial value for finding the largest contour
					best_cnt = np.array([[0,0]]) # set initial value for best contour
					
					for contour in contours: # iterate over the list of contours

						area = cv2.contourArea(contour) # find the area

						if self.area_min <= area <= self.area_max and area > max_area: # if the area is within the range and greater than the current max area...
							max_area = area # set the new max area
							best_cnt = contour # store the contour

					if best_cnt.shape[0] >= 3: # if contour is more than 3 pixels

						# find centroid using "Moments" method (based on center of mass)
						M = cv2.moments(best_cnt)
						mcx,mcy = (M["m10"]/M["m00"]), (M["m01"]/M["m00"])
						cx,cy = int(mcx),int(mcy)
						
						#get orientation using SVD
						#major, minor, o1, o2 = get_orientation(best_cnt)

						draw_color = (0,255,0)

						cv2.drawContours(frame, [best_cnt], -1, draw_color, thickness = 1)

						#cv2.circle(frame,(cx,cy), 40, bullseye_color, 1)
						#cv2.circle(frame,(cx,cy), 30, bullseye_color, 1)
						#cv2.circle(frame,(cx,cy), 20, bullseye_color, 1)
						#cv2.circle(frame,(cx,cy), 20, draw_color, 1)
						cv2.circle(frame,(cx,cy), 2, draw_color, -1)
					
						vanishing_trail[:vtrail_len-2] = vanishing_trail[2:]
						vanishing_trail[-1] = np.array([cx,cy])

						for point in vanishing_trail:
							cv2.circle(frame, (int(point[0]), int(point[1])), 2, draw_color, -1)
				else:
					vanishing_trail = np.ones((vtrail_len,2))*-1
						#cv2.arrowedLine(frame, tuple(major[0].astype(np.int16)), tuple(major[1].astype(np.uint16)), bullseye_color, 2, tipLength = 0.5)

				if data_output != None:
					if type(self.source) == str:
						write_str = str(cap.get(cv2.CAP_PROP_POS_MSEC)) + "," + str(frame_number) + "," + str(mcx) + "," + str(mcy) + "\n"
					elif type(self.source) == int:
						write_str = str(time) + "," + str(frame_number) + "," + str(mcx) + "," + str(mcy) + "\n"
					self.savefile.write(write_str)
				
				if show_bg == True:
					bg_img = fgbg.getBackgroundImage()
					frame = np.concatenate((frame, bg_img), axis = 1)
				if show_mask == True:
					mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
					frame = np.concatenate((frame, mask), axis = 1)
				
				cv2.imshow("Tracker",frame)

				if video_output != None:
					out.write(frame)
								
				k = cv2.waitKey(int(self.delay*1000)) & 0xff
				if k == 27:
					break
			elif frame_number == total_frames:
				break
		 
		cap.release()
		cv2.destroyWindow("Tracker")

		if data_output != None:
			self.savefile.close()     
		if video_output != None:
			out.release()

		for i in range(10):
			cv2.waitKey(1)

def get_orientation(contour):
	contour = contour.reshape(contour.shape[0],2)
	x = contour[:,0]
	y = contour[:,1]

	# subtract the mean from each variable (column) in the data
	mx = x - np.mean(x)
	my = y - np.mean(y)
	data = np.array([mx,my])

	eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)

	PC1_eigen = 0.3 * eigenvalues[0]
	PC2_eigen = 0.3 * eigenvalues[1]
	PC1_x = [0,V[0,0]*PC1_eigen]+np.mean(x)
	PC1_y = [0,V[0,1]*PC1_eigen]+np.mean(y)
	PC2_x = [0,V[1,0]*PC2_eigen]+np.mean(x)
	PC2_y = [0,V[1,1]*PC2_eigen]+np.mean(y)
	
	major = np.array([PC1_x,PC1_y]).T
	minor = np.array([PC2_x,PC2_y]).T
	
	ratio = eigenvalues[0]/eigenvalues[1]

	PC1_xdiff = PC1_x[1] - PC1_x[0]
	PC1_ydiff = PC1_y[1] - PC1_y[0]
	PC2_xdiff = PC2_x[1] - PC2_x[0]
	PC2_ydiff = PC2_y[1] - PC2_y[0]

	orientation1 = np.degrees((np.arctan2(PC1_ydiff,PC1_xdiff)))
	orientation2 = np.degrees((np.arctan2(PC2_ydiff,PC2_xdiff)))
	
	return [major, minor, orientation1, orientation2]