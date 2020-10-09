# Loading the required python modules
# import pytesseract # this is tesseract module
import matplotlib.pyplot as plt
import cv2 # this is opencv module
import glob
import os
import config
import matplotlib.gridspec as gridspec

DEBUG = config.DEBUG
DEBUG_IMG_PAUSE = config.DEBUG_IMG_PAUSE


def recognize_plates(frame):
    predicted_license_plates = []

    test_license_plate = frame

    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: initial frame', test_license_plate)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    # image resizing
    resize_test_license_plate = cv2.resize(
        test_license_plate, None, fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC)

    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: resize', resize_test_license_plate)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    # converting to Gray-scale
    grayscale_resize_test_license_plate = cv2.cvtColor(
        resize_test_license_plate, cv2.COLOR_BGR2GRAY)

    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: greyscale', grayscale_resize_test_license_plate)
        cv2.waitKey(DEBUG_IMG_PAUSE)

    # denoising the Image
    gaussian_blur_license_plate = cv2.GaussianBlur(
        grayscale_resize_test_license_plate, (5, 5), 0)
    if DEBUG:
        cv2.imshow('pr_pytess_opencv_g4g: gaussian blur', gaussian_blur_license_plate)
        cv2.waitKey(DEBUG_IMG_PAUSE)




        # im = cv.imread('test.jpg')
        # imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        # ret, thresh = cv.threshold(imgray, 127, 255, 0)
        # contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


    # cv2.findContours()

#     return
#
#     img_for_pytess = gaussian_blur_license_plate
#
# #    predicted_result = pytesseract.image_to_string(img_for_pytess, config = '--psm 8, -c tessedit_char_whitelist=ABCEHIKMOPTX0123456789') #,
#                                                              #  config='--oem 0 --psm 8 -c tessedit_char_whitelist = ABCEHIKMOPTX0123456789')
#     filter_new_predicted = "".join(predicted_result.split()).replace(":", "").replace("-",
#                                                                                                                  "")
#     print('filter_new_predicted: ', filter_new_predicted)


    def nguen_preproc(lpimage):

        #lpimage = cv2.cvtColor(lpimage, cv2.COLOR_BGR2GRAY)

        plate_image = lpimage #cv2.convertScaleAbs(lpimage, alpha=(255.0))

        # Applied inversed thresh_binary
        binary = cv2.threshold(plate_image, 180, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        inverse = cv2.bitwise_not(thre_mor)

        for img, name in zip([plate_image, binary, kernel3, thre_mor, inverse],
                             ['plate_image', 'binary', 'kernel3', 'thre_mor',
                             'inverse']):
            cv2.imshow('nguen_preproc ' + name, img)
            cv2.waitKey(0)

            cont, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create sort_contours() function to grab the contour of each digit from left to right
            def sort_contours(cnts, reverse=False):
                i = 0
                boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                    key=lambda b: b[1][i], reverse=reverse))
                return cnts

            # cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # creat a copy version "test_roi" of plat_image to draw bounding box
            test_roi = plate_image.copy()

            # Initialize a list which will be used to append charater image
            crop_characters = []

            # define standard width and height of character
            digit_w, digit_h = 30, 60

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h / w
                if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
                    if h / plate_image.shape[
                        0] >= 0.5:  # Select contour which has the height larger than 50% of the plate
                        # Draw bounding box arroung digit number
                        cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Sperate number and gibe prediction
                        curr_num = thre_mor[y:y + h, x:x + w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)
                cv2.imshow(name + ' with boxes', test_roi)
                cv2.waitKey(0)

    nguen_preproc(gaussian_blur_license_plate)

            #
            #
            # img = cv2.drawContours(plate_image, cont, -1, (0,205,0), 5 )
            # cv2.imshow('draw_cout' + name, img)
            # cv2.waitKey(0)
            #
            # print(name, 'len(cont), cont:', len(cont))
            #
            #
            # print(name, pytesseract.image_to_string(img, config = '--psm 8, -c tessedit_char_whitelist=ABCEHIKMOPTX0123456789'))


        # ng_lp = nguen_preproc(gaussian_blur_license_plate)
        #
        # predicted_result_ng = pytesseract.image_to_string(ng_lp, config = '--psm 8, -c tessedit_char_whitelist=ABCEHIKMOPTX0123456789') #,
        #                                                          #  config='--oem 0 --psm 8 -c tessedit_char_whitelist = ABCEHIKMOPTX0123456789')
        # filter_new_predicted_ng = "".join(predicted_result_ng.split()).replace(":", "").replace("-",
        #                                                                                                              "")
        # print('filter_new_predicted_ng:', filter_new_predicted_ng)


    # TODO make confidense getting
    confidence = -1


    return 0 #filter_new_predicted, confidence


def recognize_plates_old(LpImg):

    LpImg = cv2.resize(LpImg, dsize=(600, 400),  fx=3, fy=3)

    LpImg = [LpImg]
    if (len(LpImg)):  # check if there is at least one license image
        # Scales, calculates absolute values, and converts the result to 8-bit.
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        # convert to grayscale and blur the image
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Applied inversed thresh_binary
        binary = cv2.threshold(blur, 180, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

    # visualize results
    fig = plt.figure(figsize=(12, 7))
    plt.rcParams.update({"font.size": 18})
    grid = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
    plot_image = [plate_image, gray, blur, binary, thre_mor]
    plot_name = ["plate_image", "gray", "blur", "binary", "dilation"]

    for i in range(len(plot_image)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.title(plot_name[i])
        if i == 0:
            plt.imshow(plot_image[i])
        else:
            plt.imshow(plot_image[i], cmap="gray")

            # plt.savefig("threshding.png", dpi=300)

    # Create sort_contours() function to grab the contour of each digit from left to right
    def sort_contours(cnts, reverse=False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return cnts

    cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()

    # Initialize a list which will be used to append charater image
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 15, 20

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h / w
        if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
            if h / plate_image.shape[0] >= 0.5:  # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Sperate number and gibe prediction
                curr_num = thre_mor[y:y + h, x:x + w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    print("Detect {} letters...".format(len(crop_characters)))
    fig = plt.figure(figsize=(10, 6))
    plt.axis(False)
    plt.imshow(test_roi)
    # plt.savefig('grab_digit_contour.png',dpi=300)




    filter_new_predicted = 'test'

    # TODO make confidense getting
    confidence = -1


    return filter_new_predicted, confidence
