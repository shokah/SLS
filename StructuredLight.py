import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob


# git password - sls123 test

def GenerateGrayCodePatternsImages(width, height, resultWidth, resultHeight,
                                   direction='vertical', layerLimit = 20):
    '''
    generate a set of binary gray code patterns in the given direction based on projector resolution
    :param width: projector width in pixels
    :param height: projector height in pixels
    :param resultWidth: screen resolution width in pixels
    :param resultHeight: screen resolution in pixels
    :param direction: vertical or horizontal lines (as string)
    :return: list of pattern images
    '''
    patterns = []
    if direction == 'vertical':
        max_lvl = int(np.log2(width))
        p = np.array([[0, 255]])
    else:
        max_lvl = int(np.log2(height))
        p = np.array([0, 255])
    patterns.append(p)
    i = 1
    max_lvl = np.min([max_lvl, layerLimit])
    # goes over each level and create the image
    while i <= max_lvl:
        cv2.imwrite(str(i)+'.jpg', p)
        img = cv2.imread(str(i)+'.jpg', 0)
        img = cv2.resize(img, (resultWidth, resultHeight))
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(i)+'.jpg', img)
        if direction == 'vertical':
            p = np.hstack((p, np.array(patterns[i-1][:, ::-1])))
        else: # not fully working, make sure to fix it
            p = np.hstack((p, np.array(patterns[i - 1][:-1])))
        patterns.append(p)
        i += 1
    return patterns


def BuildCodeMatrix(pathes, mode='pattern'):
    '''
    stack the images, the stacked matrix gives the binary code in x,y cell
    :param pathes: list of images paths
    :return: numpy array of height x width x depth(number of images)
    '''
    code_map = 0
    c = 0
    if mode == 'pattern':
        thresh_min = 127
        thresh_max = 255
    else:
        thresh_min = 100
        thresh_max = 200
    for p in pathes:  # this loops build the planes code map
        img = cv2.imread(p, 0)
        ret, img = cv2.threshold(img, thresh_min, thresh_max, cv2.THRESH_BINARY)
        # plt.imshow(img)
        # plt.show()
        if c != 0:  # other iterations
            code_map[:, :, c] = img
        else:    # first iteration
            rows, cols, depth = img.shape[0], img.shape[1], len(pathes)
            code_map = np.zeros((rows, cols, depth), dtype=int)
            code_map[:, :, c] = img
        c += 1
    code_map[code_map > 0] = 1   # make the map binary
    return code_map

def BuildLightPlanes(paths, projCalibMatrix, distCoeffs, distBetweenPlanes = 2):
    '''
    build the light planes normals that goes from the projector perspective center
    :param pathes: list of patterns paths (vertical)
    :param projCalibMatrix: projector calibration matrix
    :param distBetweenPlanes: gap between light planes
    :return:
    '''
    xp, yp, projectorFocalLength = projCalibMatrix[0, 2], projCalibMatrix[1, 2], \
                                   (projCalibMatrix[0, 0] + projCalibMatrix[1, 1]) * 0.5
    planes = {}
    mat = BuildCodeMatrix(paths)
    perspective = np.hstack((projCalibMatrix, np.reshape([0, 0, 0], (3, 1))))

    for i in xrange(0, mat.shape[1], distBetweenPlanes):
        key = "".join(map(str, mat[int(yp), i]))
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
        # old version - without distortions - keep it might need it later
        # x, y1, y2 = i - xp, -yp, yp
        # v1 = np.array([x, y1, -projectorFocalLength])
        # v2 = np.array([x, y2, -projectorFocalLength])
        n = np.cross(v1, v2)
        planes[key] = n/np.linalg.norm(n)
    return planes

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
    angle = np.degrees(np.arccos(b/(np.linalg.norm(rayDirection) * np.linalg.norm(planeNormal))))
    if 75 < angle < 105:  # in case of ray and light plane are parallel
        return None
    c = a.dot(planeNormal)
    d = c * (rayDirection)
    return rayOrigin + (d / b)

def ComputePointsCloud(lightPlanes, camCodeMatrix, projPosition,
                       camCalibMatrix, distCoeffs, camPosition,
                       camRotation, startPoint, endPoint, distanceBetweenRays = 2):
    '''
    compute all the 3d points in the scene
    :param lightPlanes: dictionary of light planes normals
    :param camCodeMatrix: binary code matrix for each pixel (height x width x depth)
    :param camCalibMatrix: camera calibration  3x3 numpy array
    :param distCoeffs: opencv distortions coeffs
    :param camPosition: camera position relative to the projector as 1x3 numpy array
    :param camRotation: camera rotation relative to the projector as 3x3 numpy array
    :param startPoint: start scan point (row, col) as tuple
    :param endPoint: end scan point (row, col) as tuple
    :return: list of XYZ points
    '''

    xp, yp, camFocalLength = camCalibMatrix[0, 2], camCalibMatrix[1, 2],\
                           (camCalibMatrix[0, 0] + camCalibMatrix[1, 1]) * 0.5
    points = []
    startRow = startPoint[0]
    endRow = endPoint[0]
    startCol = startPoint[1]
    endCol = endPoint[1]
    perspective = np.hstack((camCalibMatrix, np.reshape(camPosition, (3, 1))))
    for i in xrange(startRow, endRow, distanceBetweenRays):
        for j in xrange(startCol, endCol, distanceBetweenRays):
            # # un distort the image point
            # p = np.array([[[j, i]]], dtype=np.float32)
            # p = cv2.undistortPoints(p, camCalibMatrix, distCoeffs, R=camRotation, P=perspective)
            # x, y = p[0, 0, 0], p[0, 0, 1]
            x, y = j - xp, -(i - yp)    # without distortion fix
            rayDirection = camRotation.dot(np.array([x, y, -camFocalLength]))
            rayDirection /= np.linalg.norm(rayDirection)
            key = "".join(map(str, camCodeMatrix[i, j]))
            if int(key) == 0:   # remove background
                continue
            try:
                planeNormal = lightPlanes[key]
                p = PlaneAndRayIntersection(projPosition, camPosition, planeNormal, rayDirection)
                if p is None:
                    continue
                points.append(p)
            except:
                pass
    return points

def WritePointsToFile(points, file_name, format='txt'):
    '''
    write the cloud points to the file in the given format
    :param points: nx3 list of points
    :param file_name: name of file
    :param format: file format - support only text now
    :return: None
    '''
    text_file = open(file_name + "." + format, "w")
    for p in points:
        str1 = str(p[0]) + "," + str(p[1]) + "," + str(p[2]) + "\n"
        text_file.write(str1)
    text_file.close()

def checkBitChange(prev_key, key):
    '''
    check if the difference in bit's between the keys is 1 return true if so
    :param prev_key: previous key
    :param key: current key
    :return:  bool
    '''
    tot = 0
    for i in xrange(len(prev_key)):
        d = np.abs(int(key[i]) - int(prev_key[i]))
        tot += d
    return tot == 1


if __name__ == '__main__':
    # GenerateGrayCodePatternsImages(800, 600, 800, 600)
    # load image to search targets in it
    patterns_paths = glob.glob('.\images\pattern\*.jpg')
    box1_paths = glob.glob('.\images\simple\\box\*.jpg')
    box2_paths = glob.glob('.\images\simple\\box2\*.jpg')
    binder_paths = glob.glob('.\images\simple\\binder\*.jpg')
    dummie_paths = glob.glob('.\images\\dummie\*.jpg')

    proj_mtx = np.array([[557.32272774, 0., 434.29075766],
                         [0., 599.19032733, 290.08959004],
                         [0., 0., 1.]], dtype=np.float32)
    proj_dist = np.array([[-1.20178761e-01, 5.02083179e-03, 2.30760632e-03, 4.06847903e-02,
                           1.52240260e-05]], dtype=np.float32)
    print "building light planes..."
    planes = BuildLightPlanes(patterns_paths, proj_mtx, proj_dist, distBetweenPlanes=8)

    projPosition = np.array([0, 0, 0])

    cam_mtx = np.array([[3.52676168e+03, 0.00000000e+00, 3.01315339e+03],
                        [0.00000000e+00, 3.40613533e+03, 1.86758386e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
    cam_dist = np.array([[1.43317881e+00, -1.60413408e+01, -5.02219786e-04, 5.46041307e-02,
                          8.13172505e+01]], dtype=np.float32)
    camPosition_box = np.array([ -4.54306512, -4.84217478, 23.56067222], dtype=np.float32)
    camRotation_box = np.array([[ 9.85955787e-01,   5.38614856e-02,   1.58082660e-01],
 [ -2.16918651e-04,   9.46977857e-01,  -3.21298756e-01],
 [ -1.67006407e-01,   3.16752077e-01,   9.33689982e-01]], dtype=np.float32)

    camPosition_dum = np.array([-9.85938852,-1.99040076, 39.25014791], dtype=np.float32)
    camRotation_dum = np.array([[ 0.97130392,  0.03227768,  0.23564135],
 [-0.01083326,  0.99572424, -0.09173807],
 [-0.2375949,   0.08655278,  0.96750053]],
                               dtype=np.float32)
    camCodeMatrix = BuildCodeMatrix(dummie_paths, 'map')
    print "building camera code matrix..."
    # camCodeMatrix = BuildCodeMatrix(box1_paths, 'map')

    # full image
    startPoint = [0, 0]
    endPoint = camCodeMatrix.shape[:2]
    print "compute point cloud..."
    points = ComputePointsCloud(planes, camCodeMatrix, projPosition,
                                cam_mtx, cam_dist, camPosition_dum, camRotation_dum,
                                startPoint, endPoint, distanceBetweenRays=8)
    # points = ComputePointsCloud(planes, camCodeMatrix, projPosition,
    #                             cam_mtx, cam_dist, camPosition_box, camRotation_box,
    #                             startPoint, endPoint, distanceBetweenRays=16)
    print "write points to file..."
    WritePointsToFile(points, "dummie_undistorted_proj", format='txt')
    print "DONE!"