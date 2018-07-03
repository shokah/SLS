import cv2
import numpy as np
import StructuredLight as sl


def DeBruijn(k, n):
    """
    de Bruijn sequence for alphabet k
    and subsequences of length n.
    """
    try:
        # let's see if k can be cast to an integer;
        # if so, make our alphabet a list
        _ = int(k)
        alphabet = list(map(str, range(k)))

    except (ValueError, TypeError):
        alphabet = k
        k = len(k)

    a = [0] * k * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return "".join(alphabet[i] for i in sequence)


def GenerateDeBrujinImage(width, height, sequence, stripe_width, color_index):
    '''
    create de-brujin image for the projector
    :param width: wanted image width in pixels
    :param height: wanted image height in pixels
    :param sequence: de-brujin sequence - 5 letter sequence
    :param stripe_width: stripe width
    :param color_index: dictionary where the keys are the values of sequence nad the values of
    the dictionary are bgr colors
    :return: open cv BGR image
    '''
    img = np.zeros((height, width, 3), np.uint8)
    col = 0
    for s in sequence:
        img[:, col:col + stripe_width] = color_index[s]
        col += 2 * stripe_width
    cv2.imwrite('de_brujin.jpg', img)
    return img


def CreateDeBrujinLightPlanes(projected_image, color_index, stripe_width, projCalibMatrix,
                              distCoeffs):
    '''
    create de-brujin light plane for each de - brujin key
    :param projected_image: the image from the projector
    :param color_index: stripe color index
    :param stripe_width: width of each stripe
    :param projCalibMatrix: projector calibration matrix
    :param distCoeffs: projector distortion coefficients
    :return: dictionary (key-debrujin code, value-plane noraml)
    '''
    height, width = projected_image.shape[0], projected_image.shape[1]
    xp, yp, projectorFocalLength = projCalibMatrix[0, 2], projCalibMatrix[1, 2], \
                                   (projCalibMatrix[0, 0] + projCalibMatrix[1, 1]) * 0.5
    perspective = np.hstack((projCalibMatrix, np.reshape([0, 0, 0], (3, 1))))
    light_planes = {}

    for i in xrange(int(0.5 * stripe_width + 2 * stripe_width),
                    int(width - 0.5 * stripe_width - 2 * stripe_width),
                    int(2 * stripe_width)):
        left_stripe = projected_image[height / 2, i - 2 * stripe_width]
        right_stripe = projected_image[height / 2, i + 2 * stripe_width]
        mid_stripe = projected_image[height / 2, i]
        key = FindStripeKey(left_stripe, color_index) + \
              FindStripeKey(mid_stripe, color_index) + \
              FindStripeKey(right_stripe, color_index)
        # undistort the points that build the light plane
        # top point
        p = np.array([[[-yp, i]]], dtype=np.float32)
        p = cv2.undistortPoints(p, projCalibMatrix, distCoeffs, R=np.eye(3), P=perspective)
        x, y1 = p[0, 0, 0], p[0, 0, 1]
        v1 = np.array([x, y1, -projectorFocalLength])
        # bot point
        p = np.array([[[yp, i]]], dtype=np.float32)
        p = cv2.undistortPoints(p, projCalibMatrix, distCoeffs, R=np.eye(3), P=perspective)
        x, y2 = p[0, 0, 0], p[0, 0, 1]
        v2 = np.array([x, y2, -projectorFocalLength])
        n = np.cross(v1, v2)
        light_planes[key] = n / np.linalg.norm(n)
    return light_planes


def FindStripeKey(stripe, color_index):
    '''
    find the matching key in the color index for the left, right and middle stripe
    :param stripe: bgr color index for light stripe (plane)
    :param color_index: dictionary of color index
    :return: key (string)
    '''
    key = 'b'
    stripe_lab = np.array(stripe, dtype=np.float32)
    stripe_lab = np.reshape(stripe_lab, (1, 1, 3))
    stripe_lab = cv2.cvtColor(stripe_lab, cv2.COLOR_BGR2LAB)
    for c in color_index:
        if CompareColors(color_index[c], stripe_lab):  # if similar
            key = c
    return key


def CompareColors(pix1, pix2, similarity_thresh=3.5):
    '''
    bool, if colors are similar return true
    :param pix1: (1,1,3) bgr color
    :param pix2: (1,1,3) bgr color
    :return: bool
    '''
    deltaE = ComputeDeltaE(pix1, pix2)
    return deltaE < similarity_thresh  # if similar


def ComputeDeltaE(pix1, pix2):
    k1 = 0.045
    k2 = 0.015
    kl = kc = kh = 1
    delta_l = pix1[0][0][0] - pix2[0][0][0]
    c1 = np.sqrt(pix1[0][0][1] ** 2 + pix1[0][0][2] ** 2)
    c2 = np.sqrt(pix2[0][0][1] ** 2 + pix2[0][0][2] ** 2)
    delta_a = pix1[0][0][1] - pix2[0][0][1]
    delta_b = pix1[0][0][2] - pix2[0][0][2]
    delta_c = c1 - c2
    delta_h = np.sqrt(delta_a ** 2 + delta_b ** 2 - delta_c ** 2)
    sl = 1
    sc = 1 + k1 * c1
    sh = 1 + k2 * c2
    t1 = (delta_l / (kl * sl)) ** 2
    t2 = (delta_c / (kc * sc)) ** 2
    t3 = (delta_h / (kh * sh)) ** 2
    delta_e = np.sqrt(t1 + t2 + t3)
    return delta_e


def ComputePointsCloud(lightPlanes, image, projPosition,
                       camCalibMatrix, distCoeffs, camPosition,
                       camRotation, startPoint, endPoint, step_size=4):
    '''
    compute all the 3d points in the scene
    :param lightPlanes: dictionary of light planes normals
    :param image: camera calibration  3x3 numpy array
    :param projPosition: position of projector
    :param camCalibMatrix: camera calibration matrix (numpy array 3x3)
    :param distCoeffs: opencv distortions coeffs
    :param camPosition: camera position relative to the projector as 1x3 numpy array
    :param camRotation: camera rotation relative to the projector as 3x3 numpy array
    :param startPoint: start scan point (row, col) as tuple
    :param endPoint: end scan point (row, col) as tuple
    :return: list of XYZ points
    '''
    xp, yp, camFocalLength = camCalibMatrix[0, 2], camCalibMatrix[1, 2], \
                             (camCalibMatrix[0, 0] + camCalibMatrix[1, 1]) * 0.5
    points = []
    startRow = startPoint[0]
    endRow = endPoint[0]
    startCol = startPoint[1]
    endCol = endPoint[1]

    for i in xrange(startRow, endRow, step_size):
        for j in xrange(startCol, endCol, step_size):
            key = DecodeDeBrujinPixel(image, (i, j), color_index)
            if key.__contains__('b'):
                continue
            x, y = j - xp, -(i - yp)  # without distortion fix
            rayDirection = camRotation.dot(np.array([x, y, -camFocalLength]))
            rayDirection /= np.linalg.norm(rayDirection)
            try:
                planeNormal = lightPlanes[key]
                p = PlaneAndRayIntersection(projPosition, camPosition, planeNormal, rayDirection)
                if p is None:
                    continue
                points.append(p)
            except:
                pass
    return points


def PlaneAndRayIntersection(planeOrigin, rayOrigin, planeNormal, rayDirection):
    '''
    compute intersection point of a line ray and a plane
    :param planeOrigin: plane's origin as numpy array 1x3
    :param rayOrigin: ray's origin as numpy array 1x3
    :param planeNormal: plane's normal as numpy array 1x3
    :param rayDirection: ray's direction as numpy array 1x3
    :return: 3d point as numpy array 1x3
    '''
    a = planeOrigin - rayOrigin
    b = rayDirection.dot(planeNormal)
    angle = np.degrees(np.arccos(b / (np.linalg.norm(rayDirection) * np.linalg.norm(planeNormal))))
    if 75 < angle < 105:  # in case of ray and light plane are parallel
        return None
    c = a.dot(planeNormal)
    d = c * (rayDirection)
    return rayOrigin + (d / b)


def DecodeDeBrujinPixel(image, pixel, color_index):
    '''
    find a pixel key in the camera image corresponding to a computed light plane
    :param image: image from the camera
    :param pixel: (row, col) of testing pixel
    :param color_index: stripes color index
    :return: light plane key (string)
    '''
    r, c = pixel[0], pixel[1]
    mid_stripe = image[pixel]
    key_mid = FindStripeKey(mid_stripe, color_index)
    if key_mid == 'b':  # landed on a black pixel, no coding
        return key_mid
    try:
        # find left key
        left_stripe, row, col = CheckChangeInStripe(image, mid_stripe, r, c, 'left')
        left_stripe, _, _ = CheckChangeInStripe(image, left_stripe, row, col, 'left')
        key_left = FindStripeKey(left_stripe, color_index)

        # find right key
        right_stripe, row, col = CheckChangeInStripe(image, mid_stripe, r, c, 'right')
        right_stripe, _, _ = CheckChangeInStripe(image, right_stripe, row, col, 'right')
        key_right = FindStripeKey(right_stripe, color_index)

        # build de-brujin plane key
        key = key_left + key_mid + key_right
        return key
    except: # handle array exceeding exception
        return 'b'


def CheckChangeInStripe(image, current, row, col, direction):
    '''
    checks for a change in stripe color. search is done column wise
    :param image: image to search in
    :param current: start pixel bgr value
    :param row: start pixel row
    :param col: start pixel col
    :param direction: search direction: 'left' or 'right'
    :return:
    '''
    pix2 = np.array(current, dtype=np.float32)
    pix2 = np.reshape(pix2, (1, 1, 3))
    pix2 = cv2.cvtColor(pix2, cv2.COLOR_BGR2LAB)
    while True:
        if direction == 'left':
            col -= 1
        else:
            col += 1

        moving = image[row, col]
        pix1 = np.array(moving, dtype=np.float32)
        pix1 = np.reshape(pix1, (1, 1, 3))
        pix1 = cv2.cvtColor(pix1, cv2.COLOR_BGR2LAB)

        if not CompareColors(pix1, pix2):
            return moving, row, col


if __name__ == '__main__':
    de_bruijn = DeBruijn(5, 3)
    width = 640
    height = 400
    stripe_width = 4
    color_index = {'0': np.array([[[255, 0, 0]]], dtype=np.float32),
                   '1': np.array([[[0, 255, 0]]], dtype=np.float32),
                   '2': np.array([[[0, 0, 255]]], dtype=np.float32),
                   '3': np.array([[[255, 255, 0]]], dtype=np.float32),
                   '4': np.array([[[255, 0, 255]]], dtype=np.float32)}

    img = GenerateDeBrujinImage(width, height, de_bruijn, stripe_width, color_index)

    # change color_index color space to LAB color space for comparison
    for c in color_index:
        color_index[c] = cv2.cvtColor(color_index[c], cv2.COLOR_BGR2LAB)

    proj_mtx = np.array([[557.32272774, 0., 434.29075766],
                         [0., 599.19032733, 290.08959004],
                         [0., 0., 1.]], dtype=np.float32)
    proj_dist = np.array([[-1.20178761e-01, 5.02083179e-03, 2.30760632e-03, 4.06847903e-02,
                           1.52240260e-05]], dtype=np.float32)
    cam_mtx = np.array([[3.52676168e+03, 0.00000000e+00, 3.01315339e+03],
                        [0.00000000e+00, 3.40613533e+03, 1.86758386e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
    cam_dist = np.array([[1.43317881e+00, -1.60413408e+01, -5.02219786e-04, 5.46041307e-02,
                          8.13172505e+01]], dtype=np.float32)
    cam_pos = np.array([10, 0, 0], dtype=np.float32)
    cam_rot = np.eye(3)
    proj_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    planes = CreateDeBrujinLightPlanes(img, color_index, stripe_width, proj_mtx, proj_dist)
    cam_image = cv2.imread('test_decode.jpg')
    # testing to see if left, right color scan works
    # points = ComputePointsCloud(planes, cam_image, proj_pos,
    #                             cam_mtx, cam_dist, cam_pos,
    #                             cam_rot, (cam_image.shape[0] / 2, cam_image.shape[1] / 2),
    #                             cam_image.shape, 1)
    points = ComputePointsCloud(planes, cam_image, proj_pos,
                                cam_mtx, cam_dist, cam_pos,
                                cam_rot, (0, 0),
                                cam_image.shape, 1)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print "write points to file..."
    sl.WritePointsToFile(points, "de_brujin", format='txt')
    print "DONE!"
