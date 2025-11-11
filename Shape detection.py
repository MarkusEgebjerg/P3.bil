import cv2
import numpy as np
afvig = 25

#Load pic
kegler = cv2.imread("gang2.png", 0)
#Blur
canny = cv2.GaussianBlur(kegler, (5, 5), 0)

#Canny
#canny = cv2.Canny(gauss, 140, 150)


#Find contours
contours,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Define list of alle contour centers, and alle cone positions (top part of cone)
contour_centers = []
cone_positions = []

for contour in contours:
    # Step 4: Approximate the contour.......
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    area = cv2.contourArea(contour)
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    current_center = (cx, cy)
    contour_centers.append(current_center)

    # Draw the contour
    if len(approx) > 3:
        cv2.drawContours(canny, [approx], 0, (255, 0, 0), 1)
        #cv2.circle(canny, (cx, cy), 4, (255, 255, 255), -1)




#Loops through alle contour centers. checks if two of them are less than "Afvig" pixels away horizontally (x).
#if there are, the center highest up (biggest y) is added to cone_positions.

for i in range(0, len(contour_centers)):
    p=0
    for j in range(0, len(contour_centers)):

        if  i!=j and -afvig < (contour_centers[i][0] - contour_centers[j][0]) < afvig and contour_centers[i][1] < contour_centers[j][1]:
                cone_positions.append((contour_centers[i],contour_centers[j])) #list med to sammenhørende contours. højeste først
        if i!=j and -afvig > (contour_centers[i][0] - contour_centers[j][0]) and afvig < (contour_centers[i][0] - contour_centers[j][0]):
            p = p+1
            if p >= len(contour_centers)-1:
                cv2.circle(canny, (contour_centers[i]), 4, (255, 255, 255), -1)

for i in range(0, len(cone_positions)):
    cv2.circle(canny, (cone_positions[i][0]), 4, (255, 255, 255), -1)







for i in range(0, len(contour_centers)):
    for j in range(0, len(contour_centers)):
            if  -afvig < (contour_centers[i][0] - contour_centers[j][0]) < afvig:
                if contour_centers[i][1] < contour_centers[j][1]:
                    cone_positions.append(contour_centers[i])
                    cv2.circle(canny, (contour_centers[i][0], contour_centers[i][1]), 2, (10, 10, 10), -1)
                elif contour_centers[i][1] > contour_centers[j][1]:






print(contour_centers)
print(cone_positions)

#Show pic
cv2.imshow("canny", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()


