import numpy as np
from glumpy import gl, app, gloo, glm
import struct, math, numpy

# Model-view-projection matrix
def compute_model_view_proj(model, view, proj):
    return np.dot(np.dot(model, view), proj)
def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = numpy.array(data, dtype=numpy.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(numpy.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = numpy.array(data, copy=False)
        data = out
    length = numpy.atleast_1d(numpy.sum(data*data, axis))
    numpy.sqrt(length, length)
    if axis is not None:
        length = numpy.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M
# Ref:
# 1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
# 2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py
def compute_calib_proj(K, x0, y0, w, h, nc, fc, window_coords='y_down'):
    """
    :param K: Camera matrix.
    :param x0, y0: The camera image origin (normally (0, 0)).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: OpenGL projection matrix.
    """
    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same
    if window_coords == 'y_up':
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn], # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ]) # This row is also standard glPerspective

    # Draw the images right side up and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords
    else:
        assert window_coords == 'y_down'
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
            [0, 0, q, qn], # This row is standard glPerspective and sets near and far planes
            [0, 0, -1, 0]
        ]) # This row is also standard glPerspective
    return proj.T

def load_ply(path):
    """
    Loads a 3D mesh model from a PLY file.

    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
    'faces' (mx3 ndarray) - the latter three are optional.
    """
    f = open(path, 'r')

    n_pts = 0
    n_faces = 0
    face_n_corners = 3 # Only triangular faces are supported
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False

    # Read header
    while True:
        line = f.readline().rstrip('\n').rstrip('\r') # Strip the newline character(s)
        if line.startswith('element vertex'):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith('element face'):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith('element'): # Some other element
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split()[-1], line.split()[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split()
            if elems[-1] == 'vertex_indices' or elems[-1] == 'vertex_index':
                # (name of the property, data type)
                face_props.append(('n_corners', elems[2]))
                for i in range(face_n_corners):
                    face_props.append(('ind_' + str(i), elems[3]))
            else:
                print('Warning: Not supported face property: ' + elems[-1])
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures
    model = {}
    model['pts'] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.float)

    pt_props_names = [p[0] for p in pt_props]
    is_normal = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), np.float)

    is_texture = False
    if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
        is_texture = True
        model['texture_uv'] = np.zeros((n_pts, 2), np.float)

    formats = { # For binary format
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                      'red', 'green', 'blue', 'texture_u', 'texture_v']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

        if is_texture:
            model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
            model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

    # Load faces
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        print('Error: Only triangular faces are supported.')
                        print('Number of face corners: ' + str(val))
                        exit(-1)
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        print('Error: Only triangular faces are supported.')
                        print('Number of face corners: ' + str(int(elems[prop_id])))
                        exit(-1)
                else:
                    # print(prop)
                    prop_vals[prop[0]] = elems[prop_id]

        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])

    f.close()

    return model


def save_ply(path, pts, pts_colors=np.array([]), pts_normals=np.array([]), faces=np.array([])):
    """
    Saves a 3D mesh model to a PLY file.

    :param path: Path to the resulting PLY file.
    :param pts: nx3 ndarray
    :param pts_colors: nx3 ndarray
    :param pts_normals: nx3 ndarray
    :param faces: mx3 ndarray
    """
    pts_colors = np.array(pts_colors)
    if pts_colors.size != 0:
        assert(len(pts) == len(pts_colors))

    valid_pts_count = 0
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            valid_pts_count += 1

    f = open(path, 'w')
    f.write(
        'ply\n'
        'format ascii 1.0\n'
        #'format binary_little_endian 1.0\n'
        'element vertex ' + str(valid_pts_count) + '\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
    )
    if pts_normals.size != 0:
        f.write(
            'property float nx\n'
            'property float ny\n'
            'property float nz\n'
        )
    if pts_colors.size != 0:
        f.write(
            'property uchar red\n'
            'property uchar green\n'
            'property uchar blue\n'
        )
    if faces.size != 0:
        f.write(
            'element face ' + str(len(faces)) + '\n'
            'property list uchar int vertex_indices\n'
        )
    f.write('end_header\n')

    format_float = "{:.4f}"
    format_3float = " ".join((format_float for _ in range(3)))
    format_int = "{:d}"
    format_3int = " ".join((format_int for _ in range(3)))
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            f.write(format_3float.format(*pts[pt_id].astype(float)))
            if pts_normals.size != 0:
                f.write(' ')
                f.write(format_3float.format(*pts_normals[pt_id].astype(float)))
            if pts_colors.size != 0:
                f.write(' ')
                f.write(format_3int.format(*pts_colors[pt_id].astype(int)))
            f.write('\n')
    for face in faces:
        f.write(' '.join(map(str, map(int, [len(face)] + list(face.squeeze())))) + ' ')
        f.write('\n')
    f.close()

def save_vis(path, views, views_level=None):
    '''
    Creates a PLY file visualizing the views.

    :param path: Path to output PLY file.
    :param views: Views as returned by sample_views().
    :param views_level: View levels as returned by sample_views().
    :return: -
    '''
    # Visualization (saved as a PLY file)
    pts = []
    normals = []
    colors = []
    for view_id, view in enumerate(views):
        R_inv = np.linalg.inv(view['R'])
        pts += [R_inv.dot(-view['t']).squeeze(),
                # R_inv.dot(np.array([[0.01, 0, 0]]).T - view['t']).squeeze(),
                # R_inv.dot(np.array([[0, 0.01, 0]]).T - view['t']).squeeze(),
                # R_inv.dot(np.array([[0, 0, 0.01]]).T - view['t']).squeeze()
                ]

        normal = R_inv.dot(np.array([0, 0, 1]).reshape((3, 1)))
        normals += [normal.squeeze(),
                    # np.array([0, 0, 0]),
                    # np.array([0, 0, 0]),
                    # np.array([0, 0, 0])
                    ]

        if views_level:
            intens = (255 * views_level[view_id]) / float(max(views_level))
        else:
            intens = 255 * view_id / float(len(views))
        colors += [[intens, intens, intens],
                   # [255, 0, 0],
                   # [0, 255, 0],
                   # [0, 0, 255]
                   ]

    save_ply(path,
                   pts=np.array(pts),
                   pts_normals=np.array(normals),
                   pts_colors=np.array(colors))

def hinter_sampling(min_n_pts, radius=1):
    '''
    Sphere sampling based on refining icosahedron as described in:
    Hinterstoisser et al., Simultaneous Recognition and Homography Extraction of
    Local Patches with a Simple Linear Classifier, BMVC 2008

    :param min_n_pts: Minimum required number of points on the whole view sphere.
    :param radius: Radius of the view sphere.
    :return: 3D points on the sphere surface and a list that indicates on which
             refinement level the points were created.
    '''

    # Get vertices and faces of icosahedron
    a, b, c = 0.0, 1.0, (1.0 + math.sqrt(5.0)) / 2.0
    pts = [(-b, c, a), (b, c, a), (-b, -c, a), (b, -c, a), (a, -b, c), (a, b, c),
           (a, -b, -c), (a, b, -c), (c, a, -b), (c, a, b), (-c, a, -b), (-c, a, b)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
             (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
             (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]

    # Refinement level on which the points were created
    pts_level = [0 for _ in range(len(pts))]

    ref_level = 0
    while len(pts) < min_n_pts:
        ref_level += 1
        edge_pt_map = {} # Mapping from an edge to a newly added point on that edge
        faces_new = [] # New set of faces

        # Each face is replaced by 4 new smaller faces
        for face in faces:
            pt_inds = list(face) # List of point IDs involved in the new faces
            for i in range(3):
                # Add a new point if this edge hasn't been processed yet,
                # or get ID of the already added point.
                edge = (face[i], face[(i + 1) % 3])
                edge = (min(edge), max(edge))
                if edge not in edge_pt_map.keys():
                    pt_new_id = len(pts)
                    edge_pt_map[edge] = pt_new_id
                    pt_inds.append(pt_new_id)

                    pt_new = 0.5 * (np.array(pts[edge[0]]) + np.array(pts[edge[1]]))
                    pts.append(pt_new.tolist())
                    pts_level.append(ref_level)
                else:
                    pt_inds.append(edge_pt_map[edge])

            # Replace the current face with 4 new faces
            faces_new += [(pt_inds[0], pt_inds[3], pt_inds[5]),
                          (pt_inds[3], pt_inds[1], pt_inds[4]),
                          (pt_inds[3], pt_inds[4], pt_inds[5]),
                          (pt_inds[5], pt_inds[4], pt_inds[2])]
        faces = faces_new

    # Project the points to a sphere
    pts = np.array(pts)
    pts *= np.reshape(radius / np.linalg.norm(pts, axis=1), (pts.shape[0], 1))

    # Collect point connections
    pt_conns = {}
    for face in faces:
        for i in range(len(face)):
            pt_conns.setdefault(face[i], set()).add(face[(i + 1) % len(face)])
            pt_conns[face[i]].add(face[(i + 2) % len(face)])

    # Order the points - starting from the top one and adding the connected points
    # sorted by azimuth
    top_pt_id = np.argmax(pts[:, 2])
    pts_ordered = []
    pts_todo = [top_pt_id]
    pts_done = [False for _ in range(pts.shape[0])]

    def calc_azimuth(x, y):
        two_pi = 2.0 * math.pi
        return (math.atan2(y, x) + two_pi) % two_pi

    while len(pts_ordered) != pts.shape[0]:
        # Sort by azimuth
        pts_todo = sorted(pts_todo, key=lambda i: calc_azimuth(pts[i][0], pts[i][1]))
        pts_todo_new = []
        for pt_id in pts_todo:
            pts_ordered.append(pt_id)
            pts_done[pt_id] = True
            pts_todo_new += [i for i in pt_conns[pt_id]] # Find the connected points

        # Points to be processed in the next iteration
        pts_todo = [i for i in set(pts_todo_new) if not pts_done[i]]

    # Re-order the points and faces
    pts = pts[np.array(pts_ordered), :]
    pts_level = [pts_level[i] for i in pts_ordered]
    pts_order = np.zeros((pts.shape[0],))
    pts_order[np.array(pts_ordered)] = np.arange(pts.shape[0])
    for face_id in range(len(faces)):
        faces[face_id] = [pts_order[i] for i in faces[face_id]]


    return pts, pts_level
