import cv2
import numpy as np
import random
import string

COUNTER = 1


def load_img(path):
    return cv2.imread(path)


def gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def invert(img):
    return 255 - img


def scale_width(img, width):
    percent = width / float(img.shape[1])
    return cv2.resize(img, (width, int(img.shape[0] * percent)))


# def scale_height(img, height):
#     percent = height / float(img.shape[0])
#     return cv2.resize(img, (int(img.shape[1] * percent), height))
#
#
# def scale_width_height(img, width, height):
#     img = scale_width(img, width)
#     img = scale_height(img, height)
#     return img


def ims(img, name=''):
    global COUNTER
    if not name:
        name = ''.join(random.choices(string.ascii_lowercase, k=3))
    cv2.imshow(str(COUNTER)+' '+name, img)
    COUNTER += 1


# def filter_mask(img):
#     low_threshold = 50
#     high_threshold = 200
#     img = cv2.Canny(img, low_threshold, high_threshold)
#     count = 3
#     kernel = np.ones((3, 3), np.uint8)
#     img = cv2.dilate(img, kernel, iterations=count)
#     ims(img, 'dilate')
#     img = cv2.erode(img, kernel, iterations=count)
#     ims(img, 'erode')
#     img = invert(img)
#     return img


def find_staff(img):
    global COUNTER
    theta = np.pi / 180
    min_width = int(img.shape[1] / 2)
    low_threshold = 50
    high_threshold = 200
    kernel_size = 3
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(img, low_threshold, high_threshold)
    rho = 1
    threshold = int(min_width / 4)
    min_line_length = min_width
    max_line_gap = min_width / 2
    #ims(edges, 'edges')
    staff = cv2.HoughLinesP(edges, rho=rho, theta=theta, threshold=threshold,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    #print(len(staff))
    #pixel_margin = img.shape[0]/50
    staff = sorted(staff, key=lambda x: x[0][1])
    #print(staff)
    # to_delete = []
    # for s in staff:
    #     print(s[0])
    # for i in range(len(staff) - 1):
    #     dist = abs(staff[i][0][1] - staff[i+1][0][1])
    #     if 0 < dist < 5 and staff[i][0][1] not in to_delete:
    #         #np.delete(staff[], j, 1)
    #         print(staff[i], staff[i+1])
    #         to_delete.append(staff[i+1][0][1])
    # print(to_delete)
    # staff = [x for x in staff if x[0][1] not in to_delete]
    # for s in staff:
    #     print(s[0])
    # print(COUNTER, len(staff))
    return staff


def draw_staff(img, staff, i):
    global COUNTER
    staff_image = np.copy(img) * 0
    for line in staff:
        for x1, y1, x2, y2 in line:
            cv2.line(staff_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    img_staff = cv2.addWeighted(img, 0.8, staff_image, 1, 0)
    # ims(line_image, 'line_image')
    # ims(edges, 'edges')
    #ims(staff_image, 'staff_img')
    staff_image = cv2.putText(staff_image, str(len(staff)), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #cv2.imwrite(f'output/sounds_img_{i}_s.png', staff_image)
    #ims(img_staff, 'img_staff')


# def find_circles(img):
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.2, 100)
#     circles_image = np.copy(img) * 0
#     print(len(circles))
#     if circles is not None:
#         circles = np.round(circles[0, :]).astype("int")
#         for (x, y, r) in circles:
#             cv2.circle(circles_image, (x, y), r, (255, 0, 0), 4)
#             cv2.circle(circles_image, (x, y), 3, (255, 0, 0), -1)
#     img_circles = cv2.addWeighted(img, 0.8, circles_image, 1, 0)
#     ims(circles_image, 'circles_image')
#     ims(img_circles, 'img_circles')


def find_notes(img):
    gray = gray_scale(img)
    ret, thresh2 = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    poziome = thresh2.copy()
    (rows, cols) = poziome.shape
    poziomesize = np.int(cols / 30)
    kernel = np.ones((1, poziomesize), np.uint8)
    poziome = cv2.erode(poziome, kernel)
    #for i in range(rows):
        #if poziome[i, np.int(cols / 2)] > 253:
            #print(i)
            #cv2.line(img, (0, i), (cols, i), (0, 255, 0), 1)
    kernel = np.ones((1, 8), np.uint8) #spłaszcza poziomo
    poziome2 = cv2.erode(thresh2, kernel)
    #ims(poziome2, 'poz2')
    kernel = np.ones((8, 1), np.uint8) #spłaszcza pionowo
    poziome3 = cv2.erode(poziome2, kernel)
    #ims(poziome3, 'poz3')
    #(cnts, _) = cv2.findContours(poziome3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (cnts, _) = cv2.findContours(poziome3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    width_sum, height_sum, width_avg, height_avg = 0, 0, 0, 0
    min_aspect = 1.0
    max_aspect = 2.0
    avg_count = 0
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        if min_aspect < aspectRatio < max_aspect:
            width_sum += w
            height_sum += h
            avg_count += 1
    width_avg = width_sum / avg_count
    height_avg = height_sum / avg_count
    size_margin = 0.3
    #print(width_avg, height_avg)
    notes = []
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w) / h
        width_margin = abs(w - width_avg) / width_avg
        height_margin = abs(h - height_avg) / height_avg
        if min_aspect < aspectRatio < max_aspect and width_margin < size_margin and height_margin < size_margin:
            notes.append(c)
    return notes


def draw_notes(img, notes):
    contours_image = np.copy(img) * 0
    for c in notes:
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.drawContours(img, c, -1, (240, 0, 159), 3)
        cv2.drawContours(contours_image, c, -1, (240, 0, 159), 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        # cv2.circle(img, (cY, cX), 3, (240, 0, 159), -1)
        img[cY, cX] = (240, 0, 159)
        contours_image[cY, cX] = (240, 0, 159)
    #ims(img, 'img_contours')
    #ims(contours_image, 'contours_image')


def find_sounds(staff, notes):
    PRINT = 0
    order = ['C1', 'D1', 'E1', 'F1', 'G1', 'A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'A2']
    order_rev = order[::-1]
    notes_coords = []
    lines_coords = []
    sectors = []
    notes_result = []
    for c in notes:
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        notes_coords.append([cX, cY])
    for line in staff:
        lines_coords.append(line[0])
    lines_coords = sorted(lines_coords, key=lambda x: x[1])
    minY = lines_coords[0][1]
    maxY = lines_coords[-1][1]
    sect_dist = int((maxY - minY) / 8)
    notes_coords = sorted(notes_coords, key=lambda x: x[0])
    front_sectors = []
    for i in range(11):
        sectors.append(lines_coords[0][1] + i*sect_dist)
    for i in range(-2, 0):
        front_sectors.append(lines_coords[0][1] + i*sect_dist)
    sectors = front_sectors + sectors
    # sectors1 = []
    # for i in range(len(lines_coords) - 1):
    #     sector = [lines_coords[i][1], lines_coords[i+1][1]]
    #     sectors1.append(sector)
    # print(sectors)
    # print(sectors1)
    # large_gap = sectors[1][1] - sectors[1][0]
    # small_gap = sectors[0][1] - sectors[0][0]
    # front_large = [sectors[0][0] - large_gap, sectors[0][0]]
    # front_small = [front_large[0] - small_gap, front_large[0]]
    # back_large = [sectors[-1][1], sectors[-1][1] + large_gap]
    # back_small = [back_large[1], back_large[1] + small_gap]
    # sectors = [front_small] + [front_large] + sectors + [back_large] + [back_small]
    # for sector in sectors:
    #     sector.append((sector[0] + sector[1]) / 2)
    for note in notes_coords:
        #distances = [abs(sector[2] - note[1]) for sector in sectors]
        distances = [abs(sector - note[1]) for sector in sectors]
        mindist = min(distances)
        index = distances.index(mindist)
        #print(distances)
        #print(note[1], index, mindist, order_rev[index])
        if len(staff) == 10 or True:
            notes_result.append([order_rev[index], note[0]])
        else:
            notes_result.append(['ER', note[0]])
    #print(notes_result)
    # if PRINT == 1:
    #     for line in lines_coords:
    #         print(line)
    #     for note in notes_coords:
    #         print(note)
    return notes_result


def draw_sounds(img, sounds, folder, name, i):
    for s in sounds:
        img = cv2.putText(img, s[0], (s[1], 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #ims(img, 'sounds_img')
    path = f'output/{folder}/{name}{i}_out.png'
    cv2.imwrite(path, img)


# def gamma(image, gamma=1.0):
#     # build a lookup table mapping the pixel values [0, 255] to
#     # their adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#         for i in np.arange(0, 256)]).astype("uint8")
#
#     # apply gamma correction using the lookup table
#     return cv2.LUT(image, table)


def process(folder, name, ext, i):
    img_path = f'./input/{folder}/{name}{i}.{ext}'
    img = load_img(img_path)
    img = scale_width(img, 2000)
    # ims(img, 'scaled')
    #img = gamma(img)
    #ims(img, 'after')
    staff = find_staff(img)
    notes = find_notes(img)
    sounds = find_sounds(staff, notes)
    draw_staff(img, staff, i)
    draw_notes(img, notes)
    draw_sounds(img, sounds, folder, name, i)


def process_folder(folder, name, ext, start, end):
    for i in range(start, end+1):
        try:
            process(folder, name, ext, i)
        except:
            pass


if __name__ == '__main__':
    d1 = ['melodies', 'melody', 'png', 1, 30]
    d2 = ['zdjecia_bartek', 'd', 'jpg', 200, 214]
    d3 = ['zdjecia', 'zdjecie', 'png', 1, 6]
    d4 = ['zdjecia_bartek', 'r', 'jpg', 100, 107]
    d_list = [d1, d2, d3, d4]
    for d in d_list:
        process_folder(*d)
    cv2.waitKey()
