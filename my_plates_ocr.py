import cv2
import numpy as np
import pytesseract
import config


DEBUG = config.DEBUG_MY_PLATES_OCR
DEBUG_IMG_PAUSE = config.DEBUG_IMG_PAUSE
# TODO refactor debug print statemens

#
# frame = cv2.imread('data/images/plates_image_01.png')


def resize_to_h200(frame, target_height=200):
    height, width =frame.shape[:2]
    scale = target_height / height
    if DEBUG:
        print('frame width: ', width, 'height', height)

    target_width = int(width * scale)
    resized = cv2.resize(frame, (target_width, target_height)
                         , interpolation=cv2.INTER_CUBIC)

    return resized

def get_bw_letters(frame):

    # preserve original
    frame_org = frame.copy()
    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: initial frame', frame_org)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    #resize to height 200px
    frame = resize_to_h200(frame)

    # preprocess image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # denoising the Image
    frame = cv2.bilateralFilter(frame, 15, 75, 75)

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    frame = opening

    # perform otsu thresholding
    ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: otsu', frame)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    # getting contours
    contours, hierarchy = cv2.findContours(frame,
                                cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        print('pr_pytess_opencv_g4g: contours found:', len(contours))
        img = resize_to_h200(frame_org.copy())
        cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        cv2.imshow('pr_pytess_opencv_g4g: contours', img)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    # detected letters contours holder
    letters_conts = []

    # getting letters contours filtering by area
    if DEBUG:
        img_copy = resize_to_h200(frame_org.copy())
    for cnt in contours:
        # not larger than 7000 px nor less than 900
        # TODO fine tune limits
        if cv2.contourArea(cnt) < 900 or cv2.contourArea(cnt) > 7000:
            if DEBUG:
                cv2.drawContours(img_copy, cnt, -1, (0, 0, 255), 3)
            continue

        # another one filtering by min rectangle area
        rect = cv2.boundingRect(cnt)
        x, y, width, height = rect
#        box = cv2.boxPoints(rect)
        if width * height < 1300:
            if DEBUG:
                cv2.drawContours(img_copy, cnt, -1, (255, 0, 0), 3)
            continue
        # filter contours with wrong aspect ratio
        aspect = height / width
        if 0.5 > aspect or aspect > 6 or height < 50 or width < 15 or width > 200:
            if DEBUG:
                cv2.drawContours(img_copy, cnt, -1, (0, 255, 0), 3)
                topleft = ( x, y )
                bottright = ( (x + width), (y + height) )
                cv2.rectangle(img_copy, topleft, bottright, (0, 125, 0), 2)
                cv2.imshow('dropped contours red ar<900 or >7000, blue sq<1300, green dims out of limits  ', img_copy)
                print('width, height, aspect: ', width, height, aspect)
                cv2.waitKey(DEBUG_IMG_PAUSE)
            continue
            # TODO add filtering by mean colour in initial image

        if DEBUG:
            cv2.imshow('dropped contours red ar<900 or >7000, blue sq<1300, green h/w < 1.3 or h/w >6  ', img_copy)
            cv2.waitKey(DEBUG_IMG_PAUSE)
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
        cv2.waitKey(DEBUG_IMG_PAUSE)
    out = np.full_like(frame, 255)
    out[mask == 255] = frame[mask == 255]

    if DEBUG:
        cv2.imshow('out ', out)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    frame = out

    # # rotate horizontally
    # # TODO make warp perspective by ellipses
    # # TODO as here: https://mzucker.github.io/2016/10/11/unprojecting-text-with-ellipses.html
    #
    # # join all letters contours
    # joined = np.vstack(letters_conts)
    # rect = cv2.minAreaRect(joined)
    #
    # (x,y), (w, h), ang = rect
    # print('(x,y), (w, h), ang: ', (x,y), (w, h), ang)
    #
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # img = frame.copy()
    # if DEBUG:
    #     cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    #     cv2.imshow('letters_box_over_boxes', out)
    #     cv2.waitKey(0)
    #
    # width = int(rect[1][0])
    # height = int(rect[1][1])
    #
    # src_pts = box.astype("float32")
    # # coordinate of the points in box points after the rectangle has been
    # # straightened
    # dst_pts = np.array([[0, height-1],
    #                     [0, 0],
    #                     [width-1, 0],
    #                     [width-1, height-1]], dtype="float32")
    #
    # # the perspective transformation matrix
    # M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    #
    # # directly warp the rotated rectangle to get the straightened rectangle
    # frame = cv2.warpPerspective(frame, M, (width, height))
    #
    # if DEBUG:
    #     cv2.imshow("crop_unrotated", frame)
    #     cv2.waitKey(0)

    # pad for better OCR
    frame =  cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, 255)
    if DEBUG:
        cv2.imshow("padded", frame)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    return frame


def recognize_letters(frame):
    recognized_letters = pytesseract.image_to_string(frame, config = '--psm 8, -c tessedit_char_whitelist=ABCEHIKMOPTX0123456789') #,

    print('[my_plates_ocr, recognize_letters]: recognized_letters: ', recognized_letters)
    # filter_new_predicted = "".join(predicted_result.split()).replace(":", "").replace("-",
    #                                                                                                              "")
    # print('filter_new_predicted: ', filter_new_predicted)

    return  recognized_letters


def recognize_plates(frame):

    height, width,  = frame.shape[:2]
    if DEBUG:
        print('[DEBUG, my_plates_ocr]:  frame.shape: width: ', width, 'height: ', height)
    if width < 90 or height < 20:
        if DEBUG:
            print('[DEBUG, my_plates_ocr]:  frame.shape: image too small, cant recognize')

        return '', False

    bw_letters = get_bw_letters(frame)
    recognized_letters = recognize_letters(bw_letters)
    print('[my_plates_ocr, recognize_plates]: recognized_letters: ', recognized_letters)
    rec_pl_succesfull = True

    return recognized_letters, rec_pl_succesfull

# recognize_plates(frame)