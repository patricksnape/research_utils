import numpy as np
from menpo.shape import TriMesh, TexturedTriMesh

def grid_triangulation(width, height):
    row_to_index = lambda x: x * width
    top_triangles = lambda x: np.concatenate([np.arange(row_to_index(x), row_to_index(x) + width - 1)[..., None],
                                              np.arange(row_to_index(x) + 1, row_to_index(x) + width)[..., None],
                                              np.arange(row_to_index(x + 1), row_to_index(x + 1) + width - 1)[..., None]], axis=1) 
 
    # Half edges are opposite directions
    bottom_triangles = lambda x: np.concatenate([np.arange(row_to_index(x + 1), row_to_index(x + 1) + width - 1)[..., None],
                                                 np.arange(row_to_index(x) + 1, row_to_index(x) + width)[..., None],
                                                 np.arange(row_to_index(x + 1) + 1, row_to_index(x + 1) + width)[..., None]], axis=1)
 
    trilist = []
    for k in xrange(height - 1):
        trilist.append(top_triangles(k))
        trilist.append(bottom_triangles(k))
        
    return np.concatenate(trilist)


def depth_to_trimesh(depth, texture=None):
    xs, ys = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
    points = np.concatenate([xs.ravel()[..., None], 
                             ys.ravel()[..., None], 
                             depth.ravel()[..., None]], axis=-1)
    trilist = grid_triangulation(depth.shape[1], depth.shape[0])
    if texture is not None:
        # Similar to doing a meshgrid
        tcoords = np.vstack(np.nonzero(np.ones_like(depth))).T.astype(np.float)
        # scale to [0, 1]
        tcoords = tcoords / np.array(depth.shape)
        # (s,t) = (y,x)
        tcoords = np.fliplr(tcoords)
        # move origin to top left
        tcoords[:, 1] = 1.0 - tcoords[:, 1]
        return TexturedTriMesh(points, tcoords, texture, trilist=trilist)
    else:
        return TriMesh(points, trilist=trilist)

