from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion
import plotly.graph_objects as go


def triangulate_pixels(array):
    """ Converts pixels into coordinates of triangles that cover the area of the pixels
    Args:
        array: Binary array with foreground and background pixels
    Returns:
        triangles: Numpy array of coordinates for triangles that cover the pixels.
                    Each triple of subsequent coordinates relates to a triangle.
    """

    rects = get_rectangles(array)


    #left right top bottom
    # 0     1    2    3
    top_left = rects[:,[0,2]]
    bottom_left = rects[:,[0,3]]
    bottom_right = rects[:,[1,3]]
    top_right = rects[:,[1,2]]

    triangles1 = np.ndarray((len(rects)*3,2))
    triangles1[::3,:] = top_left
    triangles1[1::3,:] = bottom_left
    triangles1[2::3,:] = bottom_right

    triangles2 = np.ndarray((len(rects)*3,2))
    triangles2[::3,:] = top_left
    triangles2[1::3,:] = top_right
    triangles2[2::3,:] = bottom_right

    triangles = np.vstack([triangles1,triangles2])

    return triangles


def zero_crossings(vector):
    """ Returns the positions of the zero crossings
    Args:
        vector: Vector with potentially positive and negative values
    Returns:
        zero_crossings: Position of the zero crossings
    """
    return np.where(np.diff(np.sign(vector)))[0]

def get_start_stop(vector):
    """ Checks for continuous segments of ones and returns start and stop coordinates
    Args:
        vector: Binary numpy array
    Returns:
        row_start_stop:
    """
    if np.sum(vector) == 0:
        return []
    row_start_stop = zero_crossings(vector-0.5)
    assert len(row_start_stop)%2 == 0
    row_start_stop = np.split(row_start_stop, len(row_start_stop)//2)
    row_start_stop = [list(item) for item in row_start_stop]
    return row_start_stop


def get_rectangles(array):
    """ Splits foreground pixels of array into rectangles by scanning row wise for changes.
    Args:
        array: Binary array of pixels
    Returns:
        rectangle: Numpy array where each row contains the coordinates in order: left, right top and bottom
    """
    ys, _ = np.where(array)
    ys = list(set(ys))#set of row-indices sorted from small to large
    ys.sort()
    ys = np.array(ys)

    old_start_stop = get_start_stop(array[ys[0],:])#e.g. [[ 6, 11], [14, 15]]
    left_right_to_top = defaultdict(dict)

    for left, right in old_start_stop:#add to left_right_to_top
        left_right_to_top[left][right] = ys[0]#smallest i.e. uppermost y

    val_range = list(range(ys[0],ys[-1]+1))[1:]
    rectangles = []
    if len(ys) > 1:
        for y in val_range:
            #print("y = ", end="")
            #print(y)
            new_start_stop = get_start_stop(array[y,:])#get start stop for next y values
            for old in old_start_stop:
                if not old in new_start_stop:#add thing
                    bottom = y - 1
                    left = old[0]
                    right = old[1]
                    top = left_right_to_top[left][right]
                    rectangles.append([left+1,right,top,bottom])
                    del(left_right_to_top[left][right])

            for new in new_start_stop:
                if not new in old_start_stop:
                    left = new[0]
                    right = new[1]
                    left_right_to_top[left][right] = y
            old_start_stop = new_start_stop

    for left, right in old_start_stop:
        bottom = ys[-1]
        top = left_right_to_top[left][right]
        rectangles.append([left+1,right,top,bottom])


    rectangle = np.array(rectangles, dtype=np.float64)
    rectangle[:,0] -= 0.5
    rectangle[:,1] += 0.5
    rectangle[:,2] -= 0.5
    rectangle[:,3] += 0.5

    return rectangle

def get_planes(tensor):
    surface = np.logical_xor(tensor,binary_erosion(tensor))

    top_plane_voxel = np.zeros(tensor.shape, dtype=np.bool)
    bottom_plane_voxel = np.zeros(tensor.shape, dtype=np.bool)
    left_plane_voxel = np.zeros(tensor.shape, dtype=np.bool)
    right_plane_voxel = np.zeros(tensor.shape, dtype=np.bool)
    back_plane_voxel = np.zeros(tensor.shape, dtype=np.bool)
    front_plane_voxel = np.zeros(tensor.shape, dtype=np.bool)

    for y, x, z in np.array(np.where(surface)).T:
        assert surface[y,x,z] == 1
        top_plane_voxel[y,x,z] = top_plane(tensor, [y,x,z])
        bottom_plane_voxel[y,x,z] = bottom_plane(tensor, [y,x,z])
        left_plane_voxel[y,x,z] = left_plane(tensor, [y,x,z])
        right_plane_voxel[y,x,z] = right_plane(tensor, [y,x,z])
        back_plane_voxel[y,x,z] = back_plane(tensor, [y,x,z])
        front_plane_voxel[y,x,z] = front_plane(tensor, [y,x,z])

    return top_plane_voxel, bottom_plane_voxel, left_plane_voxel, right_plane_voxel, back_plane_voxel, front_plane_voxel


def bottom_plane(tensor, pos):
    return tensor[pos[0],pos[1],pos[2]-1] == 0#y,x,z

def top_plane(tensor, pos):
    return tensor[pos[0],pos[1],pos[2]+1] == 0

def left_plane(tensor, pos):
    return tensor[pos[0]-1,pos[1],pos[2]] == 0

def right_plane(tensor, pos):
    return tensor[pos[0]+1,pos[1],pos[2]] == 0

def back_plane(tensor, pos):
    return tensor[pos[0],pos[1]-1,pos[2]] == 0

def front_plane(tensor, pos):
    return tensor[pos[0],pos[1]+1,pos[2]] == 0

def get_surface_voxels(tensor):
    surface = np.logical_xor(tensor,binary_erosion(tensor))

    top_voxels = np.zeros(tensor.shape, dtype=np.bool)
    bottom_voxels = np.zeros(tensor.shape, dtype=np.bool)
    left_voxels = np.zeros(tensor.shape, dtype=np.bool)
    right_voxels = np.zeros(tensor.shape, dtype=np.bool)
    back_voxels = np.zeros(tensor.shape, dtype=np.bool)
    front_voxels = np.zeros(tensor.shape, dtype=np.bool)

    for y, x, z in np.array(np.where(surface)).T:
        assert surface[y,x,z] == 1
        top_voxels[y,x,z] = top_plane(tensor, [y,x,z])
        bottom_voxels[y,x,z] = bottom_plane(tensor, [y,x,z])
        left_voxels[y,x,z] = left_plane(tensor, [y,x,z])
        right_voxels[y,x,z] = right_plane(tensor, [y,x,z])
        back_voxels[y,x,z] = back_plane(tensor, [y,x,z])
        front_voxels[y,x,z] = front_plane(tensor, [y,x,z])
    return top_voxels, bottom_voxels, left_voxels, right_voxels, back_voxels, front_voxels

def get_surfaces(tensor):
    top, bottom, left, right, back, front = get_surface_voxels(tensor)
    triangles = np.ndarray([0,3])

    _, _, zs = np.array(np.where(top))
    for current_z in list(set(zs)):#Top planes
        tri = triangulate_pixels(top[:,:,current_z].T)
        tri = np.insert(tri, 2, current_z+0.5, axis = 1)
        triangles = np.vstack([triangles, np.array(tri)])
    _, _, zs = np.array(np.where(bottom))
    for current_z in list(set(zs)):#Top planes
        tri = triangulate_pixels(bottom[:,:,current_z].T)
        tri = np.insert(tri, 2, current_z-0.5, axis = 1)
        triangles = np.vstack([triangles, np.array(tri)])

    _, xs, _ = np.array(np.where(back))
    for current_x in list(set(xs)):
        tri = triangulate_pixels(back[:,current_x,:].T)
        tri = np.insert(tri, 1, current_x-0.5, axis = 1)
        triangles = np.vstack([triangles, np.array(tri)])
    _, xs, _ = np.array(np.where(front))
    for current_x in list(set(xs)):
        tri = triangulate_pixels(front[:,current_x,:].T)
        tri = np.insert(tri, 1, current_x+0.5, axis = 1)
        triangles = np.vstack([triangles, np.array(tri)])

    ys, _, _ = np.array(np.where(right))
    for current_y in list(set(ys)):
        tri = triangulate_pixels(right[current_y,:,:].T)
        tri = np.insert(tri, 0, current_y+0.5, axis = 1)
        triangles = np.vstack([triangles, np.array(tri)])

    ys, _, _ = np.array(np.where(left))
    for current_y in list(set(ys)):
        tri = triangulate_pixels(left[current_y,:,:].T)
        tri = np.insert(tri, 0, current_y-0.5, axis = 1)
        triangles = np.vstack([triangles, np.array(tri)])

    return triangles

def voxels_to_mesh(tensor, color = "blue", opacity=0.50):
    tensor = np.pad(tensor, 1)
    x, y, z = get_surfaces(tensor).T
    n_triangles = len(y)//3
    i = np.arange(n_triangles)*3
    j = np.arange(n_triangles)*3+1
    k = np.arange(n_triangles)*3+2

    data=[go.Mesh3d(x=x, y=y, z=z,i=i,j=j,k=k, color=color, opacity=opacity, xaxis_range=xaxis_range, yaxis_range=yaxis_range]
    return data
