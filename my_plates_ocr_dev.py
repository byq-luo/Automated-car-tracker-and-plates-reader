import cv2
import numpy as np
import pytesseract
import config

DEBUG = config.DEBUG
DEBUG_IMG_PAUSE = config.DEBUG_IMG_PAUSE


image = 'data/images/my_extras/one_car_plates20.jpg'
frame = cv2.imread(image)


def find_contours_draft(platesimg):

    # resize_test_license_plate = cv2.resize(platesimg, None, fx=5, fy=5,
    #     interpolation=cv2.INTER_CUBIC)
    #
    # if DEBUG:
    #     cv2.imshow('pr_pytess_opencv_g4g: resize', resize_test_license_plate)
    #     cv2.waitKey(DEBUG_IMG_PAUSE)

    # converting to Gray-scale
    grayscale_img = cv2.cvtColor(
        platesimg, cv2.COLOR_BGR2GRAY)

    # blurring and applying closing and opening morphology
    # to get rid of unnecessary details
    gaussian_blur_license_plate = cv2.GaussianBlur(
        grayscale_img, (5, 5), 0)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(gaussian_blur_license_plate, cv2.MORPH_CLOSE, kernel3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
    ret, otsu = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    #
    # if DEBUG:
    #     cv2.imshow('pr_pytess_opencv_g4g: greyscale', grayscale_resize_test_license_plate)
    #     cv2.waitKey(DEBUG_IMG_PAUSE)
    #
    # # denoising the Image
    # gaussian_blur_license_plate = cv2.GaussianBlur(
    #     grayscale_resize_test_license_plate, (3, 3), 0)
    # if DEBUG:
    #     cv2.imshow('pr_pytess_opencv_g4g: gaussian blur', gaussian_blur_license_plate)
    #     cv2.waitKey(DEBUG_IMG_PAUSE)
    #
    #
    # ret, otsu = cv2.threshold(gaussian_blur_license_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # if DEBUG:
    #     cv2.imshow('pr_pytess_opencv_g4g: otsu', otsu)
    #     cv2.waitKey(DEBUG_IMG_PAUSE)
    #
    #
    #
    #
    # ret, thresh = cv2.threshold(gaussian_blur_license_plate, 50, 255, 0)
    # adaptive_thr = cv2.adaptiveThreshold(gaussian_blur_license_plate, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,11,2)
    # if DEBUG:
    #     cv2.imshow('pr_pytess_opencv_g4g: thresh', thresh)
    #     cv2.waitKey(DEBUG_IMG_PAUSE)
    #
    # kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # thre_mor = cv2.morphologyEx(gaussian_blur_license_plate, cv2.MORPH_DILATE, kernel3)
    # thre_mor = cv2.morphologyEx(thre_mor, cv2.MORPH_ERODE, kernel3)
    # ret, otsu = cv2.threshold(thre_mor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # if DEBUG:
    #     cv2.imshow('pr_pytess_opencv_g4g: otsu+ threemor', otsu)
    #     cv2.waitKey(DEBUG_IMG_PAUSE)
    #


    #  try mor morhpology
    gaussian_blur_license_plate = cv2.GaussianBlur(
        grayscale_img, (3, 3), 0)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(gaussian_blur_license_plate, cv2.MORPH_DILATE, kernel3)
    ret, otsu = cv2.threshold(thre_mor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: otsu+ threemor', otsu)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    #  try mor morhpology
    gaussian_blur_license_plate = cv2.bilateralFilter(grayscale_img, 15, 75, 75)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(gaussian_blur_license_plate, cv2.MORPH_DILATE, kernel3)
    ret, otsu = cv2.threshold(thre_mor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: otsu+ threemor bila', otsu)
        cv2.waitKey(DEBUG_IMG_PAUSE)




    to_more_morph = otsu.copy()

    # for i in range(3, 9, 1):

    i = 5
    for img, name in zip([opening, closing, otsu],
                         ['opening', 'closing', 'otsu']):
        if DEBUG:
            cv2.imshow('pr_pytess_opencv_g4g: blur ' + str(i) +
                       ' type: ' + name, img)
            cv2.waitKey(DEBUG_IMG_PAUSE)

    return

    for img, name in zip([grayscale_resize_test_license_plate,
                          gaussian_blur_license_plate, thresh,
                          adaptive_thr, thre_mor],
             ['grayscale_resize_test_license_plate', 'gaussian_blur_license_plate'
                 , 'thresh', 'adaptive_thr', 'thre_mor']):


        edged = cv2.Canny(img, 75, 200)

        if DEBUG:
            cv2.imshow('pr_pytess_opencv_g4g: edged after ' + name , edged)
            cv2.waitKey(DEBUG_IMG_PAUSE)

        # # detect and show lines
        # lines = cv2.HoughLinesP(img, 1, math.pi / 2, 2, None, 30, 1)
        #
        # print('lines: ', len(lines))
        #
        # for line in lines[0]:
        #     #        print('line: ', line)
        #     #        frame = resize_to_h200(frame_org)
        #     pt1 = (line[0], line[1])
        #     pt2 = (line[2], line[3])
        #     to_show = cv2.line(img, pt1, pt2, (0, 0, 255), 3)
        #
        # if DEBUG:
        #     cv2.imshow('pr_pytess_opencv_g4g: lines ' + name, to_show)
        #     cv2.waitKey(DEBUG_IMG_PAUSE)


    edged = cv2.Canny(gaussian_blur_license_plate, 75, 200)

    for img, name in zip([grayscale_resize_test_license_plate,
                          gaussian_blur_license_plate, thresh,
                          adaptive_thr, thre_mor, edged, otsu],
             ['grayscale_resize_test_license_plate', 'gaussian_blur_license_plate'
                 , 'thresh', 'adaptive_thr', 'thre_mor', 'edged', 'otsu']):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print('contours found:', len(contours))
        img_copy = resize_test_license_plate.copy()
        result_frame = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)
        cv2.imshow('result ' + name, result_frame)
        cv2.waitKey(DEBUG_IMG_PAUSE)


def resize_to_h200(frame, target_height=200):
    height, width =frame.shape[:2]
    scale = target_height / height
    print('frame width: ', width, 'height', height)

    target_width = int(width * scale)

    resized = cv2.resize(frame, (target_width, target_height)
                         , interpolation=cv2.INTER_CUBIC)

    return resized


def get_bw_letters(frame):

    # preserve original
    frame_org = frame.copy()

    #resize to height 200px
    frame = resize_to_h200(frame)

    # preprocess image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # denoising the Image
    frame = cv2.bilateralFilter(frame, 15, 75, 75)

    # morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    frame = opening

    # get otsu thresholded image
    ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: otsu', frame)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    # getting contours
    contours, hierarchy = cv2.findContours(frame,
                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        print('pr_pytess_opencv_g4g: contours found:', len(contours))
        img_copy = resize_to_h200(frame_org.copy())
        result_frame = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)
        cv2.imshow('pr_pytess_opencv_g4g: contours', frame)
        cv2.waitKey(DEBUG_IMG_PAUSE)


    # detected letters contours holder
    letters_conts = []

    # getting letters contours selecting by area
    # TODO add width/height check
    for cnt in contours:
        # not larger than 7000 px nor less than 900
        # TODO fine tune limits
        if cv2.contourArea(cnt) < 900 or cv2.contourArea(cnt) > 7000:
            continue

        # another one filtering by min rectangle area
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        if cv2.contourArea(box) < 3500:
            continue
        letters_conts.append(cnt)

    if DEBUG:
        img_copy = resize_to_h200(frame_org.copy())
        result_frame = cv2.drawContours(img_copy, letters_conts, -1, (0, 255, 0), 3)
        cv2.imshow('letters contours', result_frame)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    # cutting out letter contours from thresholded image
    # to white background
    mask = np.zeros_like(frame)
    for box in letters_conts:
        cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
    if DEBUG:
        cv2.imshow('black', mask)
        cv2.waitKey(0)
    out = np.full_like(frame, 255)
    out[mask == 255] = frame[mask == 255]

    if DEBUG:
        cv2.imshow('out ', out)
        cv2.waitKey(0)

    # rotate horizontally
    # TODO make warp perspective by ellipses
    # as here: https://mzucker.github.io/2016/10/11/unprojecting-text-with-ellipses.html

    # join all letters contours
    joined = np.vstack(letters_conts)
    rect = cv2.minAreaRect(joined)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = frame.copy()
    if DEBUG:
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        cv2.imshow('letters_box_over_boxes', out)
        cv2.waitKey(0)



    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    frame = cv2.warpPerspective(frame, M, (width, height))

    if DEBUG:
        cv2.imshow("crop_unrotated", frame)
        cv2.waitKey(0)

    # pad for better OCR
    frame =  cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, 255)
    if DEBUG:
        cv2.imshow("padded", frame)
        cv2.waitKey(0)


    return frame


def find_contours(frame):

    frame_org = frame.copy()
    grayscale_img = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY)

    # blurring and applying closing and opening morphology
    # to get rid of unnecessary details
    gaussian_blur_license_plate = cv2.bilateralFilter(grayscale_img, 15, 75, 75)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(gaussian_blur_license_plate, cv2.MORPH_CLOSE, kernel3)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel3)
    ret, otsu = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: otsu + threemor 5', otsu)
        cv2.waitKey(DEBUG_IMG_PAUSE)
    frame = otsu

    ##
    ##
    ## part 2 - bounding box around plate
    ##
    ##

    #
    # lines = cv2.HoughLines(otsu, 1, np.pi / 180, 200)
    # for rho, theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #
    # show = cv2.line(resize_to_h200(frame_org), (x1, y1), (x2, y2), (0, 0, 255), 2)
    # cv2.imshow('contours', show)
    # cv2.waitKey(DEBUG_IMG_PAUSE)
    #
    # return


    contours, hierarchy = cv2.findContours(frame,
                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('contours found:', len(contours))
    img_copy = resize_to_h200(frame_org.copy())
    result_frame = cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)
    # cv2.imshow('contours', result_frame)
    # cv2.waitKey(DEBUG_IMG_PAUSE)

    img_copy = resize_to_h200(frame_org.copy())

    # looking to find large contours and draw a convex hull

    # get plates and letters by area
    plates_cont = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 7000:
            continue
        plates_cont.append(cnt)
        img_copy = resize_to_h200(frame_org.copy())
        result_frame = cv2.drawContours(img_copy, cnt, -1, (0, 255, 0), 3)
        # cv2.imshow('plates contours', result_frame)
        # cv2.waitKey(DEBUG_IMG_PAUSE)

        epsilon = 0.2 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        img_copy = resize_to_h200(frame_org.copy())
        result_frame = cv2.drawContours(img_copy, approx, -1, (0, 255, 0), 3)
        # cv2.imshow('plates contours approx', result_frame)
        # cv2.waitKey(DEBUG_IMG_PAUSE)


    letters_conts = []
    rotated_boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 900 or cv2.contourArea(cnt) > 7000:
            continue


        img_copy = resize_to_h200(frame_org.copy())
        result_frame = cv2.drawContours(img_copy, letters_conts, -1, (0, 255, 0), 3)
        # cv2.imshow('letters contours', result_frame)
        # cv2.waitKey(DEBUG_IMG_PAUSE)

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        area = cv2.contourArea(box)

        box = np.int0(box)

        # print('box: ', box )
        #
        # cv2.drawContours(img_copy, [box],  -1, (0, 255, 0), 3)
        # cv2.putText(img_copy, 'area: ' + str(area), (80, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1 , (0, 255, 0), 3)
        # cv2.imshow('letters contours', result_frame)

        #cv2.waitKey(DEBUG_IMG_PAUSE)

        if cv2.contourArea(box) < 3500:
            continue
        letters_conts.append(cnt)
        rotated_boxes.append(box)



    # img = img_copy.copy()
    # cnts = letters_conts
    # joined = []
    # # for cnt in cnts:
    # joined = np.vstack(cnts)
    # cnt = joined
    # hull = cv2.convexHull(cnt)
    # cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)
    # cv2.imshow('letters_hull', img)
    # cv2.waitKey(0)

    img = img_copy.copy()
    cnts = rotated_boxes
    joined = []
    joined = np.vstack(cnts)
    cnt = joined
    # hull = cv2.convexHull(cnt)
    # cv2.drawContours(img, [hull], 0, (0, 0, 255), 2)
    # cv2.imshow('letters_hull', img)
    # cv2.waitKey(0)

    img = img_copy.copy()
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    #box = box[1]
    # print('box : ', box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    cv2.imshow('letters_box_over_boxes', otsu)
    cv2.waitKey(0)

    ## filling ewrth outside letter with white
    img = img

    fill_color = [127, 256, 32]  # any BGR color value to fill with
    mask_value = 255  # 1 channel white (can be any non-zero uint8 value)

    # contours to fill outside of
    contours = joined
    print('joined: ', contours)
    contours = np.int0(contours)


    img = otsu

    mask = np.zeros_like(img)
    black = np.zeros(otsu.shape)
    for box in letters_conts:
        cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)

    cv2.imshow('black', mask)
    cv2.waitKey(0)
    out = np.zeros_like(img)
    out.fill(255)
    out[mask == 255] = img[mask == 255]

    cv2.imshow('out ', out)
    cv2.waitKey(0)

    return








    # stencil = np.zeros(otsu.shape).astype(otsu.dtype)
    # print('otsu.shape: ', otsu.shape)
    # contours = joined
    # # contours = [np.array([[100, 180], [200, 280], [200, 180]]), numpy.array([[280, 70], [12, 20], [80, 150]])]
    # color = [0,0,0]
    # cv2.fillPoly(stencil, contours, color)
    # result = cv2.bitwise_and(img, stencil)
    # cv2.imshow('filled ', result)
    # cv2.waitKey(0)

    # img = img_copy.copy()
    # rows, cols = otsu.shape
    # center = rect[0]
    # angle = rect[2]
    # print('center, angle: ', center, angle)
    #
    # rot = cv2.getRotationMatrix2D(center, angle, 1)
    # print(rot)
    # img = cv2.warpAffine(otsu, rot, (rows, cols))
    # cv2.imshow('warped', img)
    # cv2.waitKey(0)


    # def crop_minAreaRect(img, rect):
    #
    #     # rotate img
    #     angle = rect[2]
    #     rows, cols = img.shape[0], img.shape[1]
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    #     img_rot = cv2.warpAffine(img, M, (cols, rows))
    #
    #     # rotate bounding box
    #     rect0 = (rect[0], rect[1], 0.0)
    #     box = cv2.boxPoints(rect0)
    #     pts = np.int0(cv2.transform(np.array([box]), M))[0]
    #     pts[pts < 0] = 0
    #
    #     # crop
    #     img_crop = img_rot[pts[1][1]:pts[0][1],
    #                pts[1][0]:pts[2][0]]
    #
    #     return img_crop


    # def subimage(image, center, theta, width, height):
    #
    #     '''
    #     Rotates OpenCV image around center with angle theta (in deg)
    #     then crops the image according to width and height.
    #     '''
    #
    #     # Uncomment for theta in radians
    #     # theta *= 180/np.pi
    #
    #     shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)
    #
    #     matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    #     image = cv2.warpAffine(src=image, M=matrix, dsize=shape)
    #
    #     x = int(center[0] - width / 2)
    #     y = int(center[1] - height / 2)
    #
    #     image = image[y:y + height, x:x + width]
    #
    #     return image
    #


    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(otsu, M, (width, height))

    cv2.imshow("crop_unrotated", warped)
    cv2.waitKey(0)

    return



    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    cv2.imshow('plates_box', img)
    cv2.waitKey(0)


    for i in [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1]:
        img_copy = resize_to_h200(frame_org.copy())

        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, i * peri, True)
            print('len approx: ', len(approx))
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                cv2.drawContours(img_copy, c, -1, (0, 255, 0), 3)
                cv2.imshow('4 edges boxes with i ' + str(i), img_copy)
                cv2.waitKey(0)
                print('4 edges: ', c)

    # looping to find large contours and draw a diagonal rectangle
    for c in contours:
        if cv2.contourArea(c) < 5000:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ## BEGIN - draw rotated rectangle
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_copy, [box], 0, (0, 191, 255), 2)
    cv2.imshow('diagonal boxes', img_copy)
    cv2.waitKey(0)





#find_contours(resize_to_h200(frame))
get_bw_letters(frame)

## to rotate
def subimage(image, center, theta, width, height):

   '''
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = int( center[0] - width/2  )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image