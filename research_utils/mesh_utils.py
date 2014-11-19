import numpy as np
from menpo.shape import TriMesh, TexturedTriMesh
from menpo.image import MaskedImage


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


def depth_to_trimesh(depth, texture=None, mask=None):
    xs, ys = np.meshgrid(range(depth.shape[1]), range(depth.shape[0]))
    # Flip the ys for the image origin change
    ys = ys.max() - ys
    points = np.concatenate([xs.ravel()[..., None],
                             ys.ravel()[..., None],
                             depth.ravel()[..., None]], axis=-1)
    trilist = grid_triangulation(depth.shape[1], depth.shape[0])
    if texture is not None:
        # Similar to doing a meshgrid
        tcoords = np.vstack(np.nonzero(np.ones_like(depth))).T.astype(np.float)
        # scale to [0, 1]
        tcoords = tcoords / np.array(depth.shape)
        # move origin to top left and flip image axis
        tcoords[:, [0, 1]] = tcoords[:, [1, 0]]
        tcoords[:, 1] = 1.0 - tcoords[:, 1]

        if texture.n_channels == 1:
            texture.pixels = np.tile(texture.pixels, [1, 1, 3])
        tm = TexturedTriMesh(points, tcoords, texture, trilist=trilist)
    else:
        tm = TriMesh(points, trilist=trilist)

    if mask is not None:
        return tm.from_mask(mask.ravel())
    else:
        return tm


def approximate_normals(depth, mask=None):
    im = MaskedImage(depth, mask=mask)
    g = im.gradient()

    normals = np.concatenate([g.masked_pixels(),
                              np.ones([g.n_true_pixels(), 1])], axis=1)
    mag = np.sqrt(np.sum(normals ** 2, axis=1))
    return np.clip(normals / mag[..., None], -1., 1.)
