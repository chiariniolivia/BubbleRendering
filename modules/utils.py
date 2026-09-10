import cv2
import numpy as np
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

from setup import *


def load_image(path: str, grayscale=True) -> np.ndarray:
    """Load an image as a numpy array from a given filepath.
    
    Parameters:
        path: file path relative to *this* file (should be relative to Bubble root directory)\\
        grayscale: boolean determining whether or not image is converted to grayscale

    Returns:
        np.ndarray of image dimension (either RGB color or single channel)    
    """
    if grayscale:
        return np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    else:
        return np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))


def pseudo_inverse(matrix: np.ndarray) -> np.ndarray:
    """Perform a pseudo inverse of a non-square matrix. 
    This is equivalent to the normal inverse for square matrices.
    
    Parameters:
        matrix: (n,m) matrix to invert
        
    Returns:
        (m,n) np.ndarray of the inverted matrix
    """
    return np.linalg.inv(matrix.T @ matrix) @ matrix.T


def QR_inverse(matrix: np.ndarray) -> np.ndarray:
    """Compute the inverse of a non-square matrix using QR decomposition.
    This should be more numerically stable than the standard pseudo inverse.
    
    Parameters:
        matrix: (n,m) matrix to invert
        
    Returns: 
        (m,n) np.ndarray of the inverted matrix
    """
    Q, R = np.linalg.qr(matrix)
    return np.linalg.inv(R) @ Q.T


###
### Loss functions
###
"""Note: the first two parameters for each of these methods is the current image and the 
reference image compared to which the loss is being calculated."""

def scale_independent_mse(image: mi.TensorXf, ref: mi.TensorXf) -> float | dr.ArrayBase:
    """Brightness-independent L2 loss function."""
    scaled_image = image / (dr.mean(dr.detach(image)) + 1e-16)
    scaled_ref = ref / (dr.mean(ref) + 1e-16)
    return dr.mean(dr.sqr(scaled_image - scaled_ref))

def mse(image: mi.TensorXf, ref: mi.TensorXf) -> float | dr.ArrayBase:
    """Normal mean-squared-error without any normalization."""
    return dr.mean(dr.sqr(image - ref))

def thresholded_mse(image: mi.TensorXf, ref: mi.TensorXf, threshold: float) -> float | dr.ArrayBase:
    """Apply a threshold to pixel values then perform mse on thresholded images."""
    i_t = dr.select(image>threshold, image, 0.0)
    r_t = dr.select(ref>threshold, ref, 0.0)
    return mse(i_t, r_t)

def invert_mse(image: mi.TensorXf, ref: mi.TensorXf) -> float | dr.ArrayBase:
    """Invert the pixel values and then perform mse"""
    return mse(1-image, 1-ref)

def masked_mse(image: mi.TensorXf, ref: mi.TensorXb, mask) -> float | dr.ArrayBase:
    """Apply mse only to region provided by the mask"""
    
    if (image.shape[:2] != mask.shape[:2]):
        print("WARNING! Mask shape does not match image shape.")

    image = dr.select(mask, image, 0)
    ref = dr.select(mask, ref, 0)

    return dr.mean(dr.sqr(image - ref))    


###
### Triangulation
###
def get_xy_scale(image_width: int = 1280, image_height: int = 800, 
                 fov: float = 100, fov_axis: str = 'x') -> np.ndarray:
    """Determine the scale factor for normalized points along the x and y axes
    to account for the fov along the provided fov axis. (Assuming no lens distortion)
    
    Parameters:
        image_width: Width of the image in pixels\\
        image_height: Height of the image in pixels\\
        fov: The field of view in degrees along axis 'fov_axis'\\
        fov_axis: The axis along which 'fov' is defined -- either 'x', or 'y'

    Returns:
        (2,) The scale factors along the x and y axes
    """
    aspect = image_width / image_height
    xscale = 1.0
    yscale = 1.0
    if fov_axis == 'x':
        xscale = np.tan(np.deg2rad(fov) / 2.0)
        yscale = np.arctan2(xscale, aspect) * xscale
    elif fov_axis == 'y':
        yscale = np.tan(np.deg2rad(fov) / 2.0)
        xscale = np.arctan(aspect * yscale) * yscale
    else:
       print("Warning! Non-valid fov-axis provided! Defaulting to 1.0 for xy scale.")
    return np.array([xscale, yscale])


def get_ideal_camera_matrix(fov: float, fov_axis: str = 'x',
                            image_width: int = 1280, 
                            image_height: int = 800) -> np.ndarray:
    """Returns the ideal camera matrix for a pinhole camera using Mitsuba's sensor size defaults.
    
    The pixel size for the Mitsuba sensor is based on dividing the fov-axis image_/height 
    by the number of pixels in that dimension.

    Parameters:
        fov: The field of view in degrees along axis 'fov_axis'\\
        fov_axis: The axis along which 'fov' is defined -- either 'x', or 'y'\\
        image_width: Width of the image in pixels\\
        image_height: Height of the image in pixels

    Returns:
        (3,3) np.ndarray representing the ideal camera matrix
    """
    if fov_axis == 'y':
        focal_length_pixels = image_height / (2 * np.tan(np.deg2rad(fov) / 2))
    else:
        focal_length_pixels = image_width / (2 * np.tan(np.deg2rad(fov) / 2))
        if fov_axis != 'x':
            print("Warning! Invalid fov_axis provided, defaulting to fov_axis = 'x'")
    
    return np.array([
        [focal_length_pixels, 0.0, image_width/2.0],
        [0.0, focal_length_pixels, image_height/2.0],
        [0.0, 0.0, 1.0]
    ])


def normalize_coordinate(pixel_coord: np.ndarray, 
                         camera_matrix: np.ndarray | None = None,
                         image_width: int = 1280, 
                         image_height: int = 800,
                         fov: float = 100,
                         fov_axis: str = 'x') -> np.ndarray:
    """Normalize (2,) UV pixel coordinate by multiplying by the inverse of the
    camera matrix. If camera matrix is not provided (i.e., `None`), compute the 
    ideal camera matrix with the provided remaining inputs.

    Parameters:
        pixel_coord: (2,) Input pixel coordinates to be normalized\\
        camera_matrix: (3,3) Camera matrix\\
        image_width: Width of the image in pixels\\
        image_height: Height of the image in pixels\\
        fov: The field of view in degrees along axis 'fov_axis'\\
        fov_axis: The axis along which 'fov' is defined -- either 'x', or 'y'

    Returns:
        (2,) The normalized UV pixel coordinates
    """
    if camera_matrix is None:
        camera_matrix = get_ideal_camera_matrix(fov, fov_axis, image_width, image_height)
    return (np.linalg.inv(camera_matrix) @ np.concatenate([pixel_coord, [1.0]]))[:2]

    # # A depracated alternate method for normalizing using the ideal camera matrix values:
    # scales = get_xy_scale(image_width, image_height, fov, fov_axis)
    # xn = (2.0 * pixel_coord[0] / image_width) - 1.0
    # yn = (2.0 * pixel_coord[1] / image_height) - 1.0
    # return np.array([xn*scales[0], yn*scales[1]])


def pixel_offset(normalized_offset: np.ndarray,
                 image_width: int = 1280,
                 image_height: int = 800,
                 fov: float = 100,
                 fov_axis: str = 'x') -> np.ndarray:
    """Take in the normalized image offset and image dimensions
    and convert the normalized offset to a pixel offset.
    This function is depracated(?)

    Parameters:
        normalized_offset: (2,) Normalized UV offset\\
        image_width: Width of the image in pixels\\
        image_height: Height of the image in pixels\\
        fov: field of view in degrees of the image\\
        fov_axis: whether the provided fov is for the 'x' or 'y' axis

    Returns:
        (2,) The corresponding UV pixel offset
    """
    scales = get_xy_scale(image_width, image_height, fov, fov_axis)

    xop = normalized_offset[0] * image_width / 2.0
    yop = normalized_offset[1] * image_height / 2.0

    return np.array([xop/scales[0], yop/scales[1]])


def normalized_to_pixel(normalized_offset: np.ndarray,
                        camera_matrix: np.ndarray | None = None,
                        image_width: int = 1280,
                        image_height: int = 800,
                        fov: float = 100,
                        fov_axis: str = 'x') -> np.ndarray:
    """Take in the normalized image offset and image dimensions
    and convert the normalized UV coordinate to a pixel coordinate
    using the provided camera matrix. If the camera matrix is `None`,
    determine the ideal camera matrix using the remaining parameters.
    
    Parameters:
        normalized_offset: (2,) Normalized UV offset\\
        camera_matrix: (3,3) Camera matrix\\
        image_width: Width of the image in pixels\\
        image_height: Height of the image in pixels\\
        fov: field of view in degrees of the image\\
        fov_axis: whether the provided fov is for the 'x' or 'y' axis

    Returns:
        (2,) The corresponding UV pixel offset
    """
    if camera_matrix is None:
        camera_matrix = get_ideal_camera_matrix(fov, fov_axis, image_width, image_height)
    return (camera_matrix @ np.concatenate([normalized_offset, [1.0]]))[:2]

    # # A depracated alternate method for computing the pixel coordinates 
    # # using the ideal camera matrix values.
    # offset = pixel_offset(normalized_offset, image_width, 
    #                       image_height, fov, fov_axis)
    # return offset + np.array([image_width//2, image_height//2])


def compute_inverse_pose_matrices(poses: np.ndarray) -> np.ndarray:
    """Take in a stack of pose matrices and compute their inverses
    by concatenating a row of [0, 0, 0, 1], taking the inverse, then returning
    the first 3 rows of the inverted matrix.
    
    Params:
        poses: (3,4,n) Stack of pose matrices

    Returns:
        inv_poses, a (3,4,n) stack of inverted pose matrices
    """
    invposes = []

    for i in range(poses.shape[0]):
        # Convert back to 4x4
        fxf = np.concatenate([poses[i], np.array([[0.0, 0.0, 0.0, 1.0]])])
        # Compute inverse and take first 3 rows
        invposes.append(np.linalg.inv(fxf)[:3,:])

    return np.stack(invposes)


def mid_point_two_view_triangulate(pose1: np.ndarray, pose2: np.ndarray, 
                                   p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Perform mid-point two-view triangulation given two pose matrices and 
    the normalized pixel coordinates of the bubbles.

    Parameters:
        pose1: (3,4) pose matrix for the first camera\\
        pose2: (3,4) pose matrix for the second camera\\
        p1: (2,) Normalized UV coordinate of the bubble in the first camera\\
        p2: (2,) Normalized UV coordinate of the bubble in the second camera

    Returns:
        (3,) World space position of estimated midpoint triangulation
    """
    # Extract the rotation matrix and translation vector from the pose matrices
    rmat1 = pose1[:,:3]
    rmat2 = pose2[:,:3]

    # Determine the out-going ray direction
    # Note: We are using -1 for the homogeneous coordinate 
    # bc Mitsuba uses left-handed coordinates!
    ray1d = rmat1 @ np.array([p1[0], p1[1], -1.0]).T
    ray2d = rmat2 @ np.array([p2[0], p2[1], -1.0]).T

    # The origin of the rays is the translation vectors of the pose matrices
    ray1o = pose1[:,3]
    ray2o = pose2[:,3]

    # Solve for the distance along each ray where they are closest to each other
    A = np.array([[ray1d.T @ ray1d, -ray2d.T @ ray1d],
                  [ray1d.T @ ray2d, -ray2d.T @ ray2d]])
    b = np.array([(ray2o - ray1o).T @ ray1d, (ray2o - ray1o).T @ ray2d])
    ts = np.linalg.inv(A) @ b

    # Find and return the point of bisection of the closest points of intersection
    r1t = ray1o + (ts[0] * ray1d)
    r2t = ray2o + (ts[1] * ray2d)
    return 0.5 * (r1t + r2t)


def n_view_triangulate(poses: np.ndarray, locs: np.ndarray) -> np.ndarray:
    """Compute the least squares optimized position for n-views by minimizing
    the distance of the predicted location to each of the rays. In our case,
    n should always be 3 since we have only 3 cameras.
    
    Parameters:
        poses: (n,3,4) Stacked pose matrices of n cameras\\
        locs: (n,2) Stacked normalized UV true locations of the bubble in each camera

    Returns:
        (3,) Least squares optimized world coordinate of the bubble
    """
    n = poses.shape[0] # number of views (should be 3 in our case)
    A = np.zeros((3, 3))
    b = np.zeros((n))

    # Iterate through each of the views to fill in A and b
    for i in range(n):

        # Extract the rotation matrix and translation vector from the pose matrix
        rmat = poses[i,:,:3]

        # Determine the normalized out-going ray direction
        # Note: We are using -1 for the homogeneous coordinate 
        # bc Mitsuba uses left-handed coordinates!
        rayn = normalize(rmat @ np.array([[locs[i,0], locs[i,1], -1.0]]).T).reshape((3,1))

        # The origin of the rays is the translation vectors of the pose matrices
        rayo = poses[i,:,3]

        # Add to A, b
        n2i = rayn @ rayn.T
        A += n2i - np.eye(3)
        b += (n2i - np.eye(3)) @ rayo.T

    return pseudo_inverse(A) @ b
    # return QR_inverse(A) @ b


def reproject(inv_pose: np.ndarray, loc: np.ndarray) -> np.ndarray:
    """Take in a (3,4) inverse pose matrix and a (3,) world space location and reproject
    that world space location to normalized UV coordinates.

    Parameters:
        inv_pose: (3,4) Inverse pose matrix\\
        loc:  (3,) World space location ([x, y, z])

    Returns:
        (2,) Normalized UV coordinate of reprojected point
    """
    # Add homogeneous coordinate to world loc and multiply by projection matrix
    rep = inv_pose @ np.concatenate([loc, [1.0]])

    # Divide by (negative) last element then truncate
    # -- We divide by negative here because Mitsuba uses left-handed coordinates.
    # If our pose matrix is provided with right-handed coordinates, we would 
    # divide by the positive value of the last element.
    return (rep / -rep[-1])[:-1]


def reprojection_error(inv_pose: np.ndarray, pred: np.ndarray,
                       loc: np.ndarray) -> np.ndarray:
    """Compute the difference between the normalized reprojected point 
    and normalized true point. (i.e., compute reprojection error in
    camera coordinates, not image coordinates)

    Parameters:
        inv_pose: (3,4) inverse pose matrix\\
        pred: (3,)  Predicted world coordinate ([x, y, z])\\
        loc:  (2,)  Actual normalized UV coordinate

    Returns:
        (2,) vector representing the difference in the normalized UV coordinates
    """
    return loc - reproject(inv_pose, pred)


def compute_combined_reprojection_error(inv_poses: np.ndarray, locs: np.ndarray,
                                        pred: np.ndarray) -> float:
    """Computes the squared length of all reprojection errors which is used as our loss.
    Note: the inputs should be stacked numpy arrays such that poses[0] returns the
    pose matrix of the first camera, etc.
    
    Parameters:
        inv_poses: (3,3,4) Stacked inverse pose matrices of the cameras\\
        locs:  (3,2)   Stacked normalized UV true locations of bubble in each camera\\
        pred:  (3,)    Predicted world coordinate ([x, y, z])

    Returns:
        Squared length of the concatenated reprojection errors (||r||^2)
    """
    errors = []
    for i in range(inv_poses.shape[0]):
        rep = reprojection_error(inv_poses[i], pred, locs[i])
        errors.append(rep)

    # Note that we technically don't need to reshape since sum will just sum over all elements
    errors = np.stack(errors).reshape(1,-1)

    return np.sum(errors**2)


def average_two_view_triangulate(poses: np.ndarray, locs: np.ndarray) -> np.ndarray:
    """Perform two view mid-point triangulations for each pair of cameras and return the average
    of the predicted locations. This method is a good initial estimate for the bubble locations.

    Parameters:
        poses: (3,3,4) Stacked pose matrices of each camera\\
        locs:  (3,2)   Stacked normalized UV coordinate of the bubble for each camera

    Returns:
        (3,) Average of the triangulated world coordinates
    """
    # Determine the initial predicted bubble location from each pair of cameras
    c12pred = mid_point_two_view_triangulate(poses[0], poses[1], locs[0], locs[1])
    c13pred = mid_point_two_view_triangulate(poses[0], poses[2], locs[0], locs[2])
    c23pred = mid_point_two_view_triangulate(poses[1], poses[2], locs[1], locs[2])

    # Determine the average of the predicted locations (this is our initial prediction)
    return ((c12pred + c13pred + c23pred) / 3.0)


def gradient_descent_optimization(poses: np.ndarray, locs: np.ndarray, 
                                  iterations: int = 20, 
                                  delta: float = 0.01,
                                  init_mode: int = 1,
                                  scale_delta: bool = True,
                                  basic_mode: bool = True) -> np.ndarray:
    """Perform gradient descent to optimize the localization of a single bubble using 
    all three cameras' pose matrices and the normalized UV coordinates of the 
    bubble in each of the three images. Note that this gradient descent optionally 
    scales the delta by the number of remaining iterations.

    Parameters:
        poses: (3,3,4) Stacked pose matrices of each camera\\
        locs:  (3,2)   Stacked normalized UV coordinate of the bubble for each camera\\
        iterations: Number of optimization iterations\\
        delta: Step size to check in each direction for each iteration\\
        init_mode: Choose whether initial guess should be made using
            0: average_two_view_triangulate;
            1: n_view_triangulate\\
        scale_delta: Boolean to determine whether delta should be scaled by the 
            number of remaining iteration steps.\\
        basic_mode: Boolean if true chooses cardinal direction. Else moves along
            computed gradient.
        

    Returns:
        (3,) Optimized world space bubble location\\
        List of (3,) past predicted locations during optimization\\
        List of floats representing losses at each iteration of optimization
    """
    # Determine the initial prediction
    pred = np.zeros((3))
    if init_mode == 1:
        pred = n_view_triangulate(poses, locs)
    else:
        pred = average_two_view_triangulate(poses, locs)

    # Store all predicted locations (for debugging / access later)
    preds = [pred]

    # Compute inverse pose matrices
    inv_poses = compute_inverse_pose_matrices(poses)

    # Store the list of losses (for debugging / access later)
    losses = [compute_combined_reprojection_error(inv_poses, locs, pred)]

    # Main optimization loop
    for i in range(iterations):
        cdelta = delta
        if scale_delta: # scale delta by number of remaining iterations
            cdelta = delta*((iterations-i)/iterations) 

        if basic_mode:
            # Test each of the 6 directions and compute their reprojection losses.
            x1 = pred + np.array([-cdelta, 0.0, 0.0])
            x1l = compute_combined_reprojection_error(inv_poses, locs, x1)
            x2 = pred + np.array([cdelta, 0.0, 0.0])
            x2l = compute_combined_reprojection_error(inv_poses, locs, x2)

            y1 = pred + np.array([0.0, -cdelta, 0.0])
            y1l = compute_combined_reprojection_error(inv_poses, locs, y1)
            y2 = pred + np.array([0.0, cdelta, 0.0])
            y2l = compute_combined_reprojection_error(inv_poses, locs, y2)

            z1 = pred + np.array([0.0, 0.0, -cdelta])
            z1l = compute_combined_reprojection_error(inv_poses, locs, z1)
            z2 = pred + np.array([0.0, 0.0, cdelta])
            z2l = compute_combined_reprojection_error(inv_poses, locs, z2)

            # Pick out the direction that minimizes the loss and update the predicted location.
            posns = [x1, x2, y1, y2, z1, z2]
            ls = [x1l, x2l, y1l, y2l, z1l, z2l]
            min_index = np.argmin(ls)
            pred = posns[min_index]
            loss = ls[min_index]
        else:
            grad = compute_gradient_of_reprojection_errors(inv_poses, locs, pred, cdelta)
            pred -= grad
            loss = compute_combined_reprojection_error(inv_poses, locs, pred)

        preds.append(pred)
        losses.append(loss)

    return pred, preds, losses


def compute_gradient_of_reprojection_errors(
        inv_poses: np.ndarray, locs: np.ndarray, 
        cpred: np.ndarray, delta: float) -> np.ndarray:
    """Compute the gradient of the combined reprojection errors if we 
    move the predicted location by some delta in each of the 3 cardinal directions
    [x - delta x -> x + detla x, y - delta y -> y + delta y, z - delta z -> z + delta z].

    Parameters:
        inv_poses: (3,3,4) Stacked inverse pose matrices of each camera\\
        locs:  (3,2) Stacked normalized UV coordinate of the bubble for each camera\\
        cpred: (3,) Current predicted world coordinate of the bubble\\
        delta: Step size to check in each direction for each iteration

    Returns:
        (3,) gradient along x, y, z numerically evaluated using a step size delta
    """
    x1 = cpred + np.array([-delta, 0.0, 0.0])
    x1l = compute_combined_reprojection_error(inv_poses, locs, x1)
    x2 = cpred + np.array([delta, 0.0, 0.0])
    x2l = compute_combined_reprojection_error(inv_poses, locs, x2)
    xl = x2l - x1l

    y1 = cpred + np.array([0.0, -delta, 0.0])
    y1l = compute_combined_reprojection_error(inv_poses, locs, y1)
    y2 = cpred + np.array([0.0, delta, 0.0])
    y2l = compute_combined_reprojection_error(inv_poses, locs, y2)
    yl = y2l - y1l

    z1 = cpred + np.array([0.0, 0.0, -delta])
    z1l = compute_combined_reprojection_error(inv_poses, locs, z1)
    z2 = cpred + np.array([0.0, 0.0, delta])
    z2l = compute_combined_reprojection_error(inv_poses, locs, z2)
    zl = z2l - z1l

    return np.array([xl, yl, zl]) / (2.0 * delta)


def compute_hessian_of_reprojection_errors(inv_poses: np.ndarray, locs: np.ndarray,
                                           cpred: np.ndarray, delta: float) -> np.ndarray:
    """Compute the Hessian matrix (matrix of second order partial derivatives) of
    the total reprojection errors using finite differences.
    
    Parameters:
        inv_poses: (3,3,4) Stacked inverse pose matrices of each camera\\
        locs:  (3,2) Stacked normalized UV coordinate of the bubble for each camera\\
        cpred: (3,) Current predicted world coordinate of the bubble\\
        delta: Step size to check in each direction for each iteration

    Returns:
        (3,3) Hessian matrix of second order partial derivatives
    """
    hessian = np.zeros((3,3))
    ijk = np.array([[delta, 0.0, 0.0], [0.0, delta, 0.0], [0.0, 0.0, delta]])

    for n, m in np.ndindex(hessian.shape):
        if hessian[n,m] != 0.0:
            p0 = cpred + ijk[m] + ijk[n]
            l0 = compute_combined_reprojection_error(inv_poses, locs, p0)
            p1 = cpred - ijk[m] + ijk[n]
            l1 = compute_combined_reprojection_error(inv_poses, locs, p1)
            p2 = cpred + ijk[m] - ijk[n]
            l2 = compute_combined_reprojection_error(inv_poses, locs, p2)
            p3 = cpred - ijk[m] - ijk[n]
            l3 = compute_combined_reprojection_error(inv_poses, locs, p3)
            h = l0 - l1 - l2 + l3
            hessian[m,n] = h
            hessian[n,m] = h
        else: # take advantage of the fact that H_mn = H_nm
            pass

    denom = 4.0 * delta * delta
    return hessian / denom


def levenberg_marquardt_optimization(poses: np.ndarray, locs: np.ndarray, 
                                     iterations: int = 100, 
                                     delta: float = 0.01,
                                     lmb: float = 0.1,
                                     scale_lmb: float = 1.0,
                                     init_mode: int = 1) -> np.ndarray:
    """Use the Levenberg-Marquardt algorithm to optimize the localization of a single 
    bubble using all three camera's projection matrices and the normalized UV coordinates 
    of the bubble in each of the three images.

    Parameters:
        poses: (3,3,4) Stacked pose matrices of each camera\\
        locs:  (3,2)   Stacked normalized UV coordinate of the bubble for each camera\\
        iterations: Number of optimization iterations\\
        delta: Step size to check in each direction for each iteration\\
        lmb: lambda, controls the amount of influence the gradient has on each step\\
        scale_lmb: The amount to scale lambda by if loss increases or the inverse 
            if loss decreases. If set to 1.0, no scaling takes place.\\
        init_mode: Choose whether initial guess should be made using
            0: average_two_view_triangulate;
            1: n_view_triangulate

    Returns:
        (3,) Optimized world space bubble location\\
        List of (3,) past predicted locations during optimization\\
        List of floats representing losses at each iteration of optimization
    """
    # Determine the initial prediction
    pred = np.zeros((3))
    if init_mode == 1:
        pred = n_view_triangulate(poses, locs)
    else:
        pred = average_two_view_triangulate(poses, locs)

    # Store all predicted locations (for debugging / access later)
    preds = [pred]

    # Compute inverse projection matrices
    inv_poses = compute_inverse_pose_matrices(poses)

    # Store the list of losses (for debugging / access later)
    losses = [compute_combined_reprojection_error(inv_poses, locs, pred)]

    # Main optimization loop
    for i in range(iterations):
        grad = compute_gradient_of_reprojection_errors(inv_poses, locs, pred, delta)
        hessian = compute_hessian_of_reprojection_errors(inv_poses, locs, pred, delta)
        pred = pred - np.linalg.inv(hessian + np.eye(3)*lmb) @ grad
        preds.append(pred)
        losses.append(compute_combined_reprojection_error(inv_poses, locs, pred))
        if losses[-1] < losses[-2]:
            # If loss decreases, reduce lambda
            lmb *= (1.0/scale_lmb)
        else:
            # Else, retract the step, increase lambda
            try:
                preds.pop()
                losses.pop()
                lmb *= scale_lmb
            except IndexError as e:
                lmb *= scale_lmb
                pass

    return pred, preds, losses


###
### Grid creation
###
default_cuboid_grid_params = {
    'xlength': 10.0, # length in cm along between end point bubbles along x-axis
    'ylength': 10.0,
    'zlength': 0.0,
    'xcount': 11, # number of bubbles in the grid along the x-axis
    'ycount': 11,
    'zcount': 1,
    'origin': [0.0, 0.0, -10.0], # origin of bubble grid
}

def generate_cuboid_grid(xlength: float, ylength: float, zlength: float, 
                         xcount: int, ycount: int, zcount: int, 
                         origin: list | np.ndarray) -> list:
    """Generate a list containing 3 element lists enumerating the xyz 
    positions of points on cuboid grid determined by the provided lengths and counts 
    along each axis and the origin.

    Parameters:
        xlength: the length in world space units the points will be distributed along the x-axis\\
        ylength: '' along the y-axis\\
        zlength: '' along the z-axis\\
        xcount: the number of points along the x-axis\\
        ycount: '' along the y-axis\\
        zcount: '' along the z-axis\\
        origin: [x, y, z] or (3,) world-space coordinates of the center of the cuboid region

    Returns:
        A list of world coordinate points that form the grid
    """

    xdelta = xlength / xcount
    xstart = origin[0] - (xdelta * (xcount - 1) / 2.0)

    ydelta = ylength / ycount
    ystart = origin[1] - (ydelta * (ycount - 1) / 2.0)

    zdelta = zlength / zcount
    zstart = origin[2] - (zdelta * (zcount - 1) / 2.0)

    grid = []
    for i in range(xcount):
        for j in range(ycount):
            for k in range(zcount):
                xposn = xstart + (i * xdelta)
                yposn = ystart + (j * ydelta)
                zposn = zstart + (k * zdelta)
                grid.append([xposn, yposn, zposn])
    
    return grid


default_cylindrical_grid_params = {
    'radius': 11.0,
    'height': 14.0,
    'rcount': 8,
    'mtcount': 45,
    'hcount':  5,
    'origin': [0.0, 0.0, -12.0],
}

def generate_cylindrical_grid(radius: float, height: float, 
                              rcount: int, mtcount: int, hcount:int, 
                              origin: list | np.ndarray) -> list:
    """Generate a cylindrical grid of points using the provided radius, height, origin,
    radial count (number of rings), max theta count (maximum points along each radial 
    ring), and height count.

    Note: to approximate an even distribution of points, the theta count is scaled 
    proportionally to the radius of each ring. Therefore the number of grid points will
    likely not be the product of rcount, tcount, and hcount!

    Parameters:
        radius: the radial extent of the outer-most ring(s) of the grid\\
        height: the z-axis extent of the grid\\
        rcount: the number of 'rings' that make up the radial extent\\
        mtcount: the number of points along the outer-most ring(s)\\
        hcount: the number of 'layers' in the grid\\
        origin: [x, y, z] or (3,) world-space coordinates of the center of the cuboid region

    Returns:
        A list of world coordinate points that form the grid
    """
    rdelta = radius / rcount
    hdelta = height / (hcount - 1 + 1e-16)
    horigin = origin[2] - (hdelta * (hcount - 1) / 2.0)

    grid = []
    for k in range(hcount):
        z = horigin + k * hdelta
        for j in range(rcount+1):
            tcount = int(mtcount * (j * rdelta / radius))
            if tcount > 0:
                tdelta = 2.0 * np.pi / tcount
                cr = j * rdelta
                for i in range(tcount):
                    x = cr * np.cos(i * tdelta)
                    y = cr * np.sin(i * tdelta)
                    grid.append([x, y, z])
            else: # central point in each layer
                grid.append([0.0, 0.0, z])
    return grid


def visualize_grid(grid: list) -> None:
    """Use matplotlib to visualize the generated grid.
    
    Parameters:
        grid: a list of grid points in format [x, y, z]
        
    Returns:
        None, only displays the resulting grid in a 3D scatter plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    for point in grid:
        ax.scatter(point[0], point[1], point[2], s=8.0)
    fig.tight_layout()


def visualize_grid_2d(grid: list) -> None:
    """Use matplotlib to visualize the generated grid compressed to
    just 2 dimensions.
    
    Parameters:
        grid: a list of grid points in format [x, y, z] or [x, y]
        
    Returns:
        None, only displays the resulting grid in a 2D scatter plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    for point in grid:
        ax.scatter(point[0], point[1], s=32.0)
    fig.tight_layout()


###
### Remapping
###
def generate_remapping_components(use_distorted_jar: bool = True) -> tuple[dict, dict]:
    """Loads in the scene components needed to create the remapping renders.
    
    Parameters:
        use_distorted_jar: a boolean indicating whether the components should be
            be updated to use the distorted jar surface(s)
            
    Returns:
        A tuple of dictionaries with the first containing the components without 
        refractive surfaces and the other with them. This corresponds to creating
        renders with true and distorted bubble locations.
    """
    # Load in scene components
    materials = create_materials(with_fluids=True)
    components0 = load_components(materials, use_distorted_jar=use_distorted_jar)
    components1 = components0.copy()

    # Remove all refractive surfaces from components0
    components0.pop('viewports_outer', None)
    components0.pop('viewports_inner', None)
    
    components0.pop('outer_jar_outer_surface', None)
    components0.pop('outer_jar_outer_surface_top', None)
    components0.pop('outer_jar_outer_surface_bottom', None)
    components0.pop('outer_jar_inner_surface', None)
    components0.pop('outer_jar_inner_surface_top', None)
    components0.pop('outer_jar_inner_surface_bottom', None)

    components0.pop('inner_jar_outer_surface', None)
    components0.pop('inner_jar_outer_surface_top', None)
    components0.pop('inner_jar_outer_surface_bottom', None)
    components0.pop('inner_jar_inner_surface', None)
    components0.pop('inner_jar_inner_surface_top', None)
    components0.pop('inner_jar_inner_surface_bottom', None)

    return components0, components1


remapping_default_config = {
    # The emitter settings for the 'bubbles' -- affects visibility in the renders
    'emitter': {
        'type':'area',
        'radiance': {
            'type': 'spectrum',
            'value': 60.0
        },
    },
    
    # The radii of the 'bubbles' in cm -- affects visibility in the renders
    'bubble_scale': 0.75,

    # The threshold used to improve circle detection 
    # -- pixel values less than this will be omitted in the hough circle transform
    'threshold': 0.99,

    # Circle detection settings
    'hough_params': {
        'method': cv2.HOUGH_GRADIENT_ALT,
        'dp': 1.5,
        'minDist': 10,
        'param1': 150,
        'param2': 0.9,
        'minRadius': 2,
        'maxRadius': 20
    },

    # Determine whether each render should be saved as a black and white image
    'save_mono': False,

    # Determine if the image renders should be denoised
    'denoise': False,
}

def rendered_grid_projections(components: dict, sensor: dict, grid: list, 
                              config: dict) -> np.ndarray:
    """Using the scene components (with or without the refractive elements),
    generate a render for each point in the grid to determine where that point
    gets mapped to. Return a list of the pixel positions.
    
    Parameters:
        components: a dict representing the Mitsuba scene\\
        sensor: a dict representing the Mitsuba sensor being used to render the scene\\
        grid: the grid of 3D points for which pixel positions will be determined\\
        config: a dict of extra params -- see 'remapping_default_config' for more info
        
    Returns:
        (n, 2) array containing corresponding pixel locations in the scene for 
        n grid points. Warning, the array may contains rows of Nones!
    """
    locations = []

    for point in grid: 
        # Add the bubble to the scene
        components.update({
            'bubble': mi.load_dict({
                'type': 'sphere',
                'focused-emitter': config['emitter'],
                'to_world': mi.ScalarTransform4f.translate(point).scale(config['bubble_scale']),
            })
        })

        # render the scene
        scene = load_scene(components=components, sensor=sensor)
        render_ = render(scene, denoise=config['denoise'], save_mono=config['save_mono'])[:,:,0]

        # Determine the projected bubble location using hough circles
        trender = np.where(render_>config['threshold'], render_, 0).astype(np.uint8)
        circles = cv2.HoughCircles(trender, **config['hough_params'])
        if circles is not None:
            locations.append(circles.reshape(-1, 3)[0][:2])
        else:
            locations.append(np.array([None, None]))

    return np.array(locations)


def remove_remapping_nones(locations0: np.ndarray, 
                           locations1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Since the Hough circle transform is not gauranteed to return a bubble location,
    this function cleans up the arrays of detected locations to make sure corresponding
    points in both arrays have valid pixel locations.

    Parameters:
        locations0: (n, 3) array of 2D pixel coordinates possibly with rows that are None\\
        locations1: (n, 3) array of 2D pixel coordinates possibly with rows that are None
    
    Returns:
        A tuple of arrays of pixel coordinates without None elements
    """
    # If the location is None in either array of locations, remove the element
    # from this array and remove the corresponding element in the other array
    Nones0 = [i for i in range(locations0.shape[0]) if np.any(locations0[i] == None)]
    for j in range(len(Nones0)):
        locations0 = np.delete(locations0, (Nones0[j]), axis=0)
        locations1 = np.delete(locations1, (Nones0[j]), axis=0)
        Nones0 = [i-1 for i in Nones0]

    Nones1 = [i for i in range(locations1.shape[0]) if np.any(locations1[i] == None)]
    for j in range(len(Nones1)):
        locations0 = np.delete(locations0, (Nones1[j]), axis=0)
        locations1 = np.delete(locations1, (Nones1[j]), axis=0)
        Nones1 = [i-1 for i in Nones1]

    return locations0, locations1


def plot_remapping_arrows(locations0: list | np.ndarray, locations1: list | np.ndarray) -> None:
    """Create an arrow plot showing how points in the image were remapped
    from without to with refractive surfaces.

    Parameters:
        locations0: a list of 2D pixel coordinates representing the projected 
            points without refractive surfaces\\
        locations1: a list of 2D pixel coordinates representing the projected
            points *with* refractive surfaces
    
    Returns:
        None; draws a matplotlib arrow plot
    """
    # Remove nones from the lists
    locations0, locations1 = remove_remapping_nones(locations0, locations1)

    # Extract the point locations...
    xp0 = []
    yp0 = []
    for loc in locations0:
        xp0.append(loc[0])
        yp0.append(loc[1])

    xp1 = []
    yp1 = []
    for loc in locations1:
        xp1.append(loc[0])
        yp1.append(loc[1])

    # Set up plot params
    plt.figure(figsize=(12.8, 8))
    plt.axis('equal')
    plt.xlim((1, 1280))
    plt.ylim((1, 800))
    plt.tight_layout()

    # Draw arrows between each corresponding pair of points
    for i in range(len(xp0)):
        plt.arrow(xp0[i], yp0[i], xp1[i]-xp0[i], yp1[i]-yp0[i], width=1.0, length_includes_head=True)


def compute_linear_distortion_coeffs(locations0: list, locations1: list) -> np.ndarray:
    """Use a least-squares approach to determine the linear remapping distortion coefficients.

    Parameters:
        locations0: a list of 2D pixel coordinates representing the projected 
            points without refractive surfaces\\
        locations1: a list of 2D pixel coordinates representing the projected
            points *with* refractive surfaces

    Returns:
        (4,1) linear distortion coefficients 
    """
    # Remove nones from the lists
    locations0, locations1 = remove_remapping_nones(locations0, locations1)

    hl = locations1.shape[0] # half length of A (half # of rows)
    A = np.zeros((2*hl, 4))
    
    i = 0 # index into row of A
    for posn in locations1:
        x = posn[0]
        y = posn[1]

        A[i,   0] = 1
        A[i,   1] = x
        A[i,   2] = 0
        A[i,   3] = 0

        A[i+1, 0] = 0
        A[i+1, 1] = 0
        A[i+1, 2] = 1
        A[i+1, 3] = y

        i += 2

    b = np.zeros((2*locations0.shape[0], 1))
    j = 0 # index into locations
    for i in range(0, 2*locations0.shape[0], 2):
        b[i  ] = locations0[j][0]
        b[i+1] = locations0[j][1]
        j += 1

    return QR_inverse(A) @ b


def linear_undistort(locs: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Appply the linear distortion coefficients to an array of 2D pixel coordinates
    to undistort their pixel coordinates.

    Parameters:
        locs: (n,2) array of 2D pixel coordinates in the format [[x0, y0], [x1, y1], ...]\\
        dist: (4,1) linear distortion coefficients

    Returns:
        (n,2) array of undistorted pixel coordinates 
    """
    undistorted = []

    for i in range(locs.shape[0]):
        if np.all(locs[i] != None):
            x = locs[i][0]
            y = locs[i][1]
            x_ = dist[0] + dist[1]*x
            y_ = dist[2] + dist[3]*y
            undistorted.append(np.array([x_, y_]))
        else:
            undistorted.append(np.array([[None], [None]]))

    return np.array(undistorted).reshape(-1,2)


def compute_polynomial_distortion_coeffs(locations0: list, locations1: list) -> np.ndarray:
    """Use a least-squares approach to determine the polynomial remapping distortion coefficients.

    Parameters:
        locations0: a list of 2D pixel coordinates representing the projected 
            points without refractive surfaces\\
        locations1: a list of 2D pixel coordinates representing the projected
            points *with* refractive surfaces

    Returns:
        (8,1) polynomial distortion coefficients 
    """
    # Remove nones from the lists
    locations0, locations1 = remove_remapping_nones(locations0, locations1)

    hl = locations1.shape[0] # half length of A (half # of rows)
    A = np.zeros((2*hl, 8))
    
    i = 0 # index into row of A
    for posn in locations1:
        x = posn[0]
        y = posn[1]

        A[i,   0] = 1
        A[i,   1] = x
        A[i,   2] = x**2
        A[i,   3] = x**4
        A[i,   4] = 0
        A[i,   5] = 0
        A[i,   6] = 0
        A[i,   7] = 0

        A[i+1, 0] = 0
        A[i+1, 1] = 0
        A[i+1, 2] = 0
        A[i+1, 3] = 0
        A[i+1, 4] = 1
        A[i+1, 5] = y
        A[i+1, 6] = y**2
        A[i+1, 7] = y**4

        i += 2

    b = np.zeros((2*locations0.shape[0], 1))
    j = 0 # index into locations
    for i in range(0, 2*locations0.shape[0], 2):
        b[i  ] = locations0[j][0]
        b[i+1] = locations0[j][1]
        j += 1

    return QR_inverse(A) @ b


def polynomial_undistort(locs: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Appply the polynomial distortion coefficients to an array of 2D pixel coordinates
    to undistort their pixel coordinates.

    Parameters:
        locs: (n,2) array of 2D pixel coordinates in the format [[x0, y0], [x1, y1], ...]\\
        dist: (8,1) polynomial distortion coefficients

    Returns:
        (n,2) array of undistorted pixel coordinates 
    """
    undistorted = []

    for i in range(locs.shape[0]):
        if np.all(locs[i] != None):
            x = locs[i][0]
            y = locs[i][1]
            x_ = dist[0] + dist[1]*x + dist[2]*(x**2) + dist[3]*(x**4)
            y_ = dist[4] + dist[5]*y + dist[6]*(y**2) + dist[7]*(y**4)
            undistorted.append(np.array([x_, y_]))
        else:
            undistorted.append(np.array([[None], [None]]))

    return np.array(undistorted).reshape(-1,2)


def compute_polynomial_extended_distortion_coeffs(locations0: list, locations1: list) -> np.ndarray:
    """Use a least-squares approach to determine the polynomial remapping distortion coefficients.
    Now upto x^8 instead of just x^4.

    Parameters:
        locations0: a list of 2D pixel coordinates representing the projected 
            points without refractive surfaces\\
        locations1: a list of 2D pixel coordinates representing the projected
            points *with* refractive surfaces

    Returns:
        (12,1) polynomial distortion coefficients 
    """
    # Remove nones from the lists
    locations0, locations1 = remove_remapping_nones(locations0, locations1)

    hl = locations1.shape[0] # half length of A (half # of rows)
    A = np.zeros((2*hl, 12))
    
    i = 0 # index into row of A
    for posn in locations1:
        x = posn[0]
        y = posn[1]

        A[i,  0] = 1
        A[i,  1] = x
        A[i,  2] = x**2
        A[i,  3] = x**4
        A[i,  4] = x**6
        A[i,  5] = x**8
        A[i,  6] = 0
        A[i,  7] = 0
        A[i,  8] = 0
        A[i,  9] = 0
        A[i, 10] = 0
        A[i, 11] = 0

        A[i+1,  0] = 0
        A[i+1,  1] = 0
        A[i+1,  2] = 0
        A[i+1,  3] = 0
        A[i+1,  4] = 0
        A[i+1,  5] = 0
        A[i+1,  6] = 1
        A[i+1,  7] = y
        A[i+1,  8] = y**2
        A[i+1,  9] = y**4
        A[i+1, 10] = y**6
        A[i+1, 11] = y**8

        i += 2

    b = np.zeros((2*locations0.shape[0], 1))
    j = 0 # index into locations
    for i in range(0, 2*locations0.shape[0], 2):
        b[i  ] = locations0[j][0]
        b[i+1] = locations0[j][1]
        j += 1

    return QR_inverse(A) @ b


def polynomial_extended_undistort(locs: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Appply the polynomial distortion coefficients to an array of 2D pixel coordinates
    to undistort their pixel coordinates.

    Parameters:
        locs: (n,2) array of 2D pixel coordinates in the format [[x0, y0], [x1, y1], ...]\\
        dist: (12,1) polynomial distortion coefficients

    Returns:
        (n,2) array of undistorted pixel coordinates 
    """
    undistorted = []

    for i in range(locs.shape[0]):
        if np.all(locs[i] != None):
            x = locs[i][0]
            y = locs[i][1]
            x_ = dist[0] + dist[1]*x + dist[2]*(x**2) + dist[3]*(x**4) + dist[ 4]*(x**6) + dist[ 5]*(x**8)
            y_ = dist[6] + dist[7]*y + dist[8]*(y**2) + dist[9]*(y**4) + dist[10]*(y**6) + dist[11]*(y**8)
            undistorted.append(np.array([x_, y_]))
        else:
            undistorted.append(np.array([[None], [None]]))

    return np.array(undistorted).reshape(-1,2)

###
### Texture creation
###
def create_rings_texture(width: int = 800, height: int = 800, npi: float = 8, 
                         min: float = 0.25, max: float = 0.75) -> None:
    """Create a cosine wave ring texture with no distortion.
    
    Parameters:
        width: texture width in pixels\\
        height: texture height in pixels\\
        npi: 2x the number of rings from the center
    
    Returns:
        Saves the texture as 'rings.png' and displays using matplotlib
    """
    from PIL import Image
    img = np.zeros((width,height))
    for j, i in np.ndindex(img.shape):
        # Normalized UV coordinates
        u = (i - width/2) / (width/2)
        v = (j - height/2) / (height/2)

        # Rings in [-1, 1] range
        img[j, i] = np.cos(8 * np.pi * np.sqrt(u*u + v*v))

    # remap range from [-1,1] to [min,max]
    hrange = max - min
    img = hrange*(0.5*(img + 1)) + min

    # Show image and save
    img = np.uint8(256*img)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    im = Image.fromarray(img)
    im = im.convert('RGB')
    im.save("rings.png")


###
### PnP Pose Estimation
###
def estimate_pose_matrix(pixel_posns: np.ndarray,
                         world_posns: np.ndarray | None = None, 
                         camera_matrix: np.ndarray | None = None,
                         distortion_coeffs: np.ndarray | None = None) -> np.ndarray:
    """Estimate the camera pose matrix using OpenCV's PnP solver.
    
    Parameters:
        pixel_posns: (5,2) the pixel positions of the markers\\
        world_posns: (5,3) corresponding world coordinates of the fiducial markers in cm. If
            None, the default posns will be used.\\
        camera_matrix: (3,3) Camera matrix. If None, uses default camera matrix.\\
        distortion_coeffs: coefficients used to undistort image positions passed into solvePnP. 
            If no distortion is present, keep as None.

    Returns:
        (3,4) estimated pose matrix (or None if solvePnP fails)
    """
    # 180 deg rotation matrix about z axis 
    # (necessary to match Mitsuba assumption about left handed coordinates)
    zflip = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

    if world_posns is None:
        world_posns = np.array([
            [-5.3029,  2.2389, 0.0],
            [-2.2382,  5.3024, 0.0],
            [ 0.0,     0.0,    0.0],
            [ 5.3031, -2.397,  0.0],
            [ 2.2381, -5.305,  0.0],
        ])

    if camera_matrix is None:
        camera_matrix = get_ideal_camera_matrix(fov=100)
    
    ret, rvec, tvec = cv2.solvePnP(world_posns, pixel_posns, camera_matrix, distortion_coeffs)
    if not ret:
        print("ERROR! solvePnP failed!")
        return None
    
    R, _ = cv2.Rodrigues(rvec)
    inv_pose = np.concatenate([zflip @ R, zflip @ tvec], axis=1)
    return np.linalg.inv(np.concatenate([inv_pose, [[0.0, 0.0, 0.0, 1.0]]], axis=0))[:3,:]


def get_cosine_angle_between_matrices(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute the cosine angle between rotation matrices P and Q.
    Let R = P @ Q.T be the difference rotation matrix. The angle is then 
    given by trace(R) = 1 + 2 cos(theta) and we solve for theta. 

    Based on: http://www.boris-belousov.net/2016/12/01/quat-dist/

    Parameters:
        P: (3,3) rotation matrix\\
        Q: (3,3) rotation matrix
    
    Returns:
        Cosine angle between the rotation matrices in degrees
    """
    R = np.dot(P, Q.T)
    cos_theta = np.clip((np.trace(R)-1)/2, -1, 1)
    return np.rad2deg(np.arccos(cos_theta))


def switch_handedness(P: np.ndarray) -> np.ndarray:
    """Multiply the (3,3) rotation or (3,4) pose matrix by the z_flip 
    matrix to convert the matrix from left to right handed coordinates 
    or vice versa.
    
    Parameters:
        P: (3,3) rotation matrix or (3,4) pose matrix

    Returns:
        Switched handedness rotation or pose matrix (or None on error)
    """
    zflip = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    if P.shape[-1] == 3:
        return zflip @ P
    elif P.shape[-1] == 4:
        R = P[:,:3]
        T = P[:,-1]
        return np.concatenate([zflip @ R, zflip @ T.reshape((-1,1))], axis=1)
    else:
        print("ERROR! Invalid shape.")
        return None


###
### Bulk Rendering
###
default_bubble_grid_config = {
    # The bubble material properties
    'bubble_material': {
        'type': 'dielectric',
        'int_ior': 1.0, # IoR of a vacuum
        'ext_ior': 1.17, # IoR of LAr
    },
    
    # The radii of the 'bubbles' in cm
    'bubble_scale': 0.5,

    # Determine whether each render should be saved as a black and white image
    'save_mono': True,

    # Determine if the image renders should be denoised
    'denoise': False,
}

def bubble_grid_renders(components: dict, sensor: dict, grid: list, 
                        config: dict = default_bubble_grid_config) -> np.ndarray:
    """Using the scene components (with or without the refractive elements),
    generate a render of a bubble for each point in the grid and save the render.
    
    Parameters:
        components: a dict representing the Mitsuba scene\\
        sensor: a dict representing the Mitsuba sensor being used to render the scene\\
        grid: the grid of 3D points for which pixel positions will be determined\\
        config: a dict of extra params -- see 'default_bubble_grid_config' for more info
        
    Returns:
        None. Instead, saves images to outputs/[date]/
        where [date] corresponds to when each render was started
    """
    for point in grid: 
        # Add the bubble to the scene
        components.update({
            'bubble': mi.load_dict({
                'type': 'sphere',
                'bsdf': config['bubble_material'],
                'to_world': mi.ScalarTransform4f.translate(point).scale(config['bubble_scale']),
            })
        })

        # render the scene
        scene = load_scene(components=components, sensor=sensor)
        render(scene, denoise=config['denoise'], save_mono=config['save_mono'])


###
### Misc
###
def random_in_unit_circle() -> np.ndarray:
    """Generate a random point in a unit circle
    
    Parameters:
        None
    
    Returns:
        (2,) array of random x, y location in unit circle
    """
    rng = np.random.default_rng()
    r = np.sqrt(rng.random())
    t = 2 * np.pi * rng.random()
    return np.array([r * np.cos(t), r * np.sin(t)])