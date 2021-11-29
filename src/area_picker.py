import cv2

def get_x_y(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print(x,y)



def pick_items():
	vid = cv2.VideoCapture('video/road_trim.mp4')
	img = vid.read()[1]
	print('1')
	while True:
		cv2.namedWindow('pick_areas')
		cv2.setMouseCallback('pick_areas', get_x_y)
		cv2.imshow('pick_areas', img)
		cv2.waitKey(10)

pick_items()