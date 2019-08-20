from demo_pb_new import CTPN
import cv2
ctpn = CTPN()


img = cv2.imread("data/demo/006.jpg")
ctpn.main2(img,"6.jpg")


img = cv2.imread("data/demo/007.jpg")
ctpn.main2(img,"7.jpg")

img = cv2.imread("data/demo/008.jpg")
ctpn.main2(img,"8.jpg")
