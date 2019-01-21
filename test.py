import cv2

imgpath='data/images/20.jpg'
txtpath = 'data/labels/20.txt'
img = cv2.imread(imgpath)
with open(txtpath, 'r') as f:
    p = f.readlines()[0]

pix = [float(h) for h in p.split()]
print(pix)

for k in range(9):
    i,j = int(pix[2*k]), int(pix[2*k+1])
    cv2.circle(img, (i,j), 3,(255,0,0), -1)
cv2.imshow('test', img)
cv2.waitKey(0)