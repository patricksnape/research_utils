import numpy as np


def fudge_ibug_68(pcloud):
    # The interior mouth points
    coincident_indices = np.array([[61, 67], [62, 66], [63, 65]])
    points = pcloud.points

    for a, b in coincident_indices:
        if np.allclose(points[a, :], points[b, :]):
            points[b, 0] += 0.1
