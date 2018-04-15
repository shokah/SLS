import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
# test
def CalibrateCamera(images, rows, cols, numOfImages=15):
    '''
    calibrate camera intristic parameters using zhang chess board calibration
    :param images: list of chess board images paths
    :param rows: number of inner corner rows in the chess board
    :param cols: number of inner corner cols in the chess board
    :param numOfImages: nu,ber of images to use in the calibration
    :return:
    '''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    c = 1
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h > w : continue
        # Resize the image to make the algorithm goes faster
        # later rescale f,xp,yp in the same way,
        # distortions are not affected by resize
        gray = cv2.resize(gray, (gray.shape[1]/2, gray.shape[0]/2))

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            # Refine the found corners to sub pixels level
            corners2 = cv2.cornerSubPix(gray, corners,
                                        (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

        if c == numOfImages:
            print c
            break
        print c
        c += 1
    # rescale the focal length and principal point positions
    resize_fix = np.array([[2, 0, 2], [0, 2, 2], [0, 0, 1]])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, resize_fix.dot(mtx), dist, rvecs, tvecs

def CalibrateProjector(imagesCamera, imagesProjector, imageChessPattern, rows, cols,
                       cameraMatrix, camDistCoeffs, numOfImages=15):
    '''
    preform a chess board calibration for a projector.
    calibration is done by taking one image of a printed board, covering it,
    then projecting a chess board and taking it's image (board surface must
     stay the same for these two images)
    with the first image the camera pose is found.
    with the camera pose and camera intrinstic parameters the projected chess
    board world coordinates
    can be computed and used for the projector calibration
    :param imagesCamera: list of camera images paths used for pose estimation
    :param imagesProjector: list of projector images paths used for calibration
    :param imageChessPattern: the projected chess pattern image (gray scale)
    :param rows: number of inner corner rows in the chess board
    :param cols: number of inner corner cols in the chess board
    :param cameraMatrix: camera calibration matrix 3x3 numpy array
    :param camDistCoeffs: camera distortion coeef
    :param numOfImages: number of images to include in the computation
    (taking the first n images)
    :return:
    '''

    # speed the algorithm - rescale camera matrix to fit the re sized shape
    cameraMatrix = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]])\
        .dot(cameraMatrix)
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    projectorImagepoints = []  # 2d points in image plane.
    projectorObjPoints = []
    c = 1

    # Resize the image to make the algorithm goes faster
    # later rescale f,xp,yp in the same way,
    # distortions are not affected by resize
    imageChessPattern = cv2.resize(imageChessPattern, (imageChessPattern.shape[1] / 2,
                                                       imageChessPattern.shape[0] / 2))
    # Find the chess board corners in the projector
    ret_chess, corners_chess = cv2.findChessboardCorners(imageChessPattern, (rows, cols), None)
    corners_chess = cv2.cornerSubPix(imageChessPattern, corners_chess, (11, 11), (-1, -1), criteria)
    corners_chess = corners_chess.astype('float32')

    for i in xrange(len(imagesCamera)):
        # this loop find camera pose per image and calculate the 3d object
        # points in projector frame
        img_cam = cv2.imread(imagesCamera[i])
        gray_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2GRAY)
        h, w = gray_cam.shape
        if h > w: continue
        # Resize the image to make the algorithm goes faster
        # later rescale f,xp,yp in the same way,
        # distortions are not affected by resize
        gray_cam = cv2.resize(gray_cam, (gray_cam.shape[1] / 2, gray_cam.shape[0] / 2))

        # Find the chess board corners
        ret_cam, corners_cam = cv2.findChessboardCorners(gray_cam, (rows, cols), None)

        img_proj = cv2.imread(imagesProjector[i])
        gray_proj = cv2.cvtColor(img_proj, cv2.COLOR_BGR2GRAY)
        h, w = gray_proj.shape
        if h > w: continue
        # Resize the image to make the algorithm goes faster
        # later rescale f,xp,yp in the same way,
        # distortions are not affected by resize
        gray_proj = cv2.resize(gray_proj, (gray_proj.shape[1] / 2, gray_proj.shape[0] / 2))

        # Find the chess board corners
        ret_proj, corners_proj = cv2.findChessboardCorners(gray_proj, (rows, cols), None)

        # If found, add object points, image points (after refining them)
        if ret_cam == True and ret_proj == True:

            # Refine the found corners to sub pixels level
            imgpoints_cam = cv2.cornerSubPix(
                gray_cam, corners_cam, (11, 11), (-1, -1), criteria)
            imgpoints_proj = cv2.cornerSubPix(
                gray_proj, corners_proj, (11, 11), (-1, -1), criteria)

            # find camera position relative to the printed chess board
            ret, rvec, tvec = cv2.solvePnP(objp, imgpoints_cam, cameraMatrix, camDistCoeffs)

            # compute object points in projector space using the image pose
            # relative to calibration board and camera calibration matrix
            R = cv2.Rodrigues(rvec)[0][:, :2]

            imgpoints_proj = cv2.undistortPoints(imgpoints_proj, cameraMatrix, camDistCoeffs)
            imgpoints_proj = imgpoints_proj.astype('float32')
            # Stack the found image points for the projector
            projectorImagepoints.append(corners_chess)
            imgpoints_proj = cv2.convertPointsToHomogeneous(imgpoints_proj)
            imgpoints_proj = np.asarray(np.reshape(imgpoints_proj,
                                                   (imgpoints_proj.shape[0],
                                                    imgpoints_proj.shape[2])),
                                        dtype=np.float32)
            K_inv = np.linalg.inv(cameraMatrix)
            Rt_inv = np.linalg.inv(np.hstack((R, tvec)))
            # calc projected chess board 3d points
            projPoints = K_inv.dot(Rt_inv).dot(imgpoints_proj.T)
            projPoints = projPoints.T
            projPoints = projPoints.astype('float32')

            projectorObjPoints.append(projPoints)
        if c == numOfImages:
            print c
            break
        print c
        c += 1
    initProjCalibMat = np.array([[imageChessPattern.shape[0], 0, imageChessPattern.shape[1]/2],
                                 [0, imageChessPattern.shape[0], imageChessPattern.shape[0]/2],
                                [0, 0, 1]], dtype=np.float32)
    # calibrate the projector based on the images of the projected chess board
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(projectorObjPoints,
                                                       projectorImagepoints,
                                                       chessboard.shape[::-1],
                                                       initProjCalibMat,
                                                       camDistCoeffs,
                                                       flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    # rescale the focal length and principal point positions
    resize_fix = np.array([[2, 0, 2], [0, 2, 2], [0, 0, 1]])
    return ret, resize_fix.dot(mtx), dist, rvecs, tvecs

def RelativeOrientation(image_cam, image_proj, rows, cols, camMatrix, camDistCoeffs, projMatrix,
                        projDistCoeffs):
    '''
    compute camera relative orientation to the projector
    :param image_cam: image of the chess board as seen by the camera
    :param image_proj: image of the projected chessboard
    :param rows: num of rows in chessboard
    :param cols: num of cols in chessboard
    :param camMatrix: camera calibration matrix
    :param camDistCoeffs: camera distortions
    :param projMatrix: projector calibration matrix
    :param projDistCoeffs: projector distortions
    :return: rotation matrix from camera to projector, translation vector from camera to projector
    '''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objectPoints = np.zeros((cols * rows, 3), np.float32)
    objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # image points for camera
    img = cv2.imread(image_cam)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        # Refine the found corners to sub pixels level
        imagePoints_cam = cv2.cornerSubPix(gray, corners,
                                    (11, 11), (-1, -1), criteria)

    # image points for projector
    img = cv2.imread(image_proj)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # Refine the found corners to sub pixels level
        imagePoints_proj = cv2.cornerSubPix(gray, corners,
                                        (11, 11), (-1, -1), criteria)

    stereocalib_retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
        cv2.stereoCalibrate([objectPoints], [imagePoints_proj], [imagePoints_cam],
                            projMatrix, projDistCoeffs, camMatrix,
                            camDistCoeffs, gray.shape[::-1], criteria=criteria,
                            flags=cv2.CALIB_FIX_INTRINSIC)
    return R, T



if __name__ == '__main__':
    images_cam = glob.glob('.\images\calibration\cam\*.jpg')
    images_proj = glob.glob('.\images\calibration\proj\*.jpg')
    chess_path = '.\images\chessboard\checkerboard.png'
    chessboard = cv2.imread(chess_path, 0)
    chessboard = cv2.resize(chessboard, (800, 600))
    rows = 6
    cols = 9
    # ret, cam_mtx, cam_dist, rvecs, tvecs = CalibrateCamera(
    #     images_cam, rows, cols, numOfImages=len(images_cam))
    # print "camera mtx ", cam_mtx
    # print 80 * '-'
    # print "camera dist ", cam_dist
    # print 80 * '-'

    cam_mtx = np.array([[3.52676168e+03, 0.00000000e+00, 3.01315339e+03],
                        [0.00000000e+00, 3.40613533e+03, 1.86758386e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    cam_dist = np.array([[1.43317881e+00, -1.60413408e+01, -5.02219786e-04, 5.46041307e-02,
                          8.13172505e+01]])

    # ret, proj_mtx, proj_dist, rvecs, tvecs = CalibrateProjector(
    #     images_cam, images_proj, chessboard, rows, cols, cam_mtx, cam_dist,
    #     numOfImages=len(images_proj))
    # print "projector mtx ", proj_mtx
    # print 80 * '-'
    # print "projector dist ", proj_dist
    # print 80 * '-'




    proj_mtx = np.array([[ 557.32272774,    0.,          434.29075766],
 [   0.,          599.19032733,  290.08959004],
 [   0.,            0.,            1.        ]])
    proj_dist = np.array([[ -1.20178761e-01,   5.02083179e-03,   2.30760632e-03,   4.06847903e-02,
    1.52240260e-05]])
    relative = images_cam[0]
    # relative = "./images/simple/relative.JPG"
    R, t = RelativeOrientation(relative,
                               chess_path,
                               rows,
                               cols,
                               cam_mtx,
                               cam_dist,
                               proj_mtx,
                               proj_dist)
    print "relative rotation matrix\n", R
    print 80 * '-'

    print "relative translation vector\n", t
    print 80 * '-'
