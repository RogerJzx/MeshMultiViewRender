import numpy as np
from glumpy import gl, app, gloo, glm
from utils import *
import cv2

## glumpy setup
# Color vertex shader
#-------------------------------------------------------------------------------
_color_vertex_code = """
uniform mat4 u_mv;
uniform mat4 u_nm;
uniform mat4 u_mvp;
uniform vec3 u_light_eye_pos;

attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_color;
attribute vec2 a_texcoord;

varying vec3 v_color;
varying vec2 v_texcoord;
varying vec3 v_eye_pos;
varying vec3 v_L;
varying vec3 v_normal;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_color = a_color;
    v_texcoord = a_texcoord;
    v_eye_pos = (u_mv * vec4(a_position, 1.0)).xyz; // Vertex position in eye coords.
    v_L = normalize(u_light_eye_pos - v_eye_pos); // Vector to the light
    v_normal = normalize(u_nm * vec4(a_normal, 1.0)).xyz; // Normal in eye coords.
}
"""

# Color fragment shader - flat shading
#-------------------------------------------------------------------------------
_color_fragment_flat_code = """
uniform float u_light_ambient_w;
uniform sampler2D u_texture;
uniform int u_use_texture;

varying vec3 v_color;
varying vec2 v_texcoord;
varying vec3 v_eye_pos;
varying vec3 v_L;

void main() {
    // Face normal in eye coords.
    vec3 face_normal = normalize(cross(dFdx(v_eye_pos), dFdy(v_eye_pos)));

    float light_diffuse_w = max(dot(normalize(v_L), normalize(face_normal)), 0.0);
    float light_w = u_light_ambient_w + light_diffuse_w;
    if(light_w > 1.0) light_w = 1.0;

    if(bool(u_use_texture)) {
        gl_FragColor = vec4(light_w * texture2D(u_texture, v_texcoord));
    }
    else {
        gl_FragColor = vec4(light_w * v_color, 1.0);
    }
}
"""



def render_rgb(model, cts, view, ssaa=4., shape=(640.480),
               K=np.array([[550, 0.0, 316.],
                           [0.0, 540., 244.],
                           [0.0, 0.0, 1.0]])):
    """
    :param model:load ply model mesh
    :param view: multi view points
    :param ssaa:defalut = 4.0
        # Super-sampling anti-aliasing (SSAA)
        # https://github.com/vispy/vispy/wiki/Tech.-Antialiasing
        # The RGB image is rendered at ssaa_fact times higher resolution and then
        # down-sampled to the required resolution.
    :param shape: rgb image shape
    :param K: camera intrisic matrix
    :return: images
    """
    assert ({'pts', 'faces'}.issubset(set(model.keys())))

    texture_uv = np.zeros((model['pts'].shape[0], 2), np.float32)
    if 'colors' in model.keys():
        assert (model['pts'].shape[0] == model['colors'].shape[0])
        colors = model['colors']
        if colors.max() > 1.0:
            colors /= 255.0  # Color values are expected in range [0, 1]
    vertices_type = [('a_position', np.float32, 3),
                     ('a_color', np.float32, colors.shape[1]),
                     ('a_texcoord', np.float32, 2)]
    vertices = np.array(zip(model['pts'],
                            colors, texture_uv), vertices_type)
    V = vertices.view(gloo.VertexBuffer)

    I = model['faces'].flatten().astype(np.uint32).view(gloo.IndexBuffer)
    program = gloo.Program(_color_vertex_code, _color_fragment_flat_code)
    program.bind(V)

    yz_flip = np.eye(4, dtype=np.float32)
    yz_flip[1, 1], yz_flip[2, 2] = -1, -1
    view = yz_flip.dot(view)  # OpenCV to OpenGL camera system
    view = view.T

    proj = compute_calib_proj(K * ssaa, 0, 0, int(shape[0] * ssaa), int(shape[1] * ssaa), 10, 10000)
    program['u_light_eye_pos'] = [0, 0, 0]  # Camera origin
    program['u_light_ambient_w'] = 0.8
    program['u_use_texture'] = int(False)
    program['u_texture'] = np.zeros((1, 1, 4), np.float32)
    model = (np.eye(4, dtype=np.float32))  ## model matrix
    mvp = compute_model_view_proj(model, view, proj)
    program['u_mvp'] = mvp
    window = app.Window(visible=False)

    points = []
    viwe_port_matrix = np.array([[shape[0]/2, 0, 0, shape[0]/2],
                           [0, -shape[1]/2, 0, shape[1]/2],
                           [0,0,0,0],
                           [0,0,0,1]])

    cts = np.concatenate((cts, np.ones((9,1), dtype=cts.dtype)), axis=1)
    for i in range(cts.shape[0]):
        p = cts[i]
        coors = viwe_port_matrix.dot(mvp.T.dot(p))
        coors/=coors[-1]
        points.append(coors[0])
        points.append(coors[1])
    global rgb
    rgb = None

    @window.event
    def on_draw(dt):
        # window.clear()
        global rgb
        extent_shape = (int(shape[0] * ssaa), int(shape[1] * ssaa))
        # Frame buffer object
        color_buf = np.zeros((extent_shape[0], extent_shape[1], 4), np.float32).view(gloo.TextureFloat2D)
        depth_buf = np.zeros((extent_shape[0], extent_shape[1]), np.float32).view(gloo.DepthTexture)
        fbo = gloo.FrameBuffer(color=color_buf, depth=depth_buf)
        fbo.activate()

        # OpenGL setup
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glViewport(0, 0, extent_shape[1], extent_shape[0])

        gl.glDisable(gl.GL_CULL_FACE)
        program.draw(gl.GL_TRIANGLES, I)

        rgb = np.zeros((extent_shape[0], extent_shape[1], 4), dtype=np.float32)
        gl.glReadPixels(0, 0, extent_shape[1], extent_shape[0], gl.GL_RGBA, gl.GL_FLOAT, rgb)
        rgb.shape = extent_shape[0], extent_shape[1], 4
        rgb = rgb[::-1, :]
        rgb = np.round(rgb[:, :, :3] * 255).astype(np.uint8)  # Convert to [0, 255]

        import cv2
        rgb = cv2.resize(rgb, shape, interpolation=cv2.INTER_AREA)

        fbo.deactivate()
    app.run(framecount=0)
    window.close()
    return rgb, points


def sample_views(min_n_views, radius=1,
                 azimuth_range=(0, 2 * math.pi),
                 elev_range=(-0.5 * math.pi, 0.5 * math.pi)):
    '''
    Viewpoint sampling from a view sphere.

    :param min_n_views: Minimum required number of views on the whole view sphere.
    :param radius: Radius of the view sphere.
    :param azimuth_range: Azimuth range from which the viewpoints are sampled.
    :param elev_range: Elevation range from which the viewpoints are sampled.
    :return: List of views, each represented by a 3x3 rotation matrix and
             a 3x1 translation vector.
    '''

    # Get points on a sphere
    if True:
        pts, pts_level = hinter_sampling(min_n_views, radius=radius)
    else:
        pts = fibonacci_sampling(min_n_views + 1, radius=radius)
        pts_level = [0 for _ in range(len(pts))]

    views = []
    for pt in pts:
        # Azimuth from (0, 2 * pi)
        azimuth = math.atan2(pt[1], pt[0])
        if azimuth < 0:
            azimuth += 2.0 * math.pi

        # Elevation from (-0.5 * pi, 0.5 * pi)
        a = np.linalg.norm(pt)
        b = np.linalg.norm([pt[0], pt[1], 0])
        elev = math.acos(b / a)
        if pt[2] < 0:
            elev = -elev

        # if hemisphere and (pt[2] < 0 or pt[0] < 0 or pt[1] < 0):
        if not (azimuth_range[0] <= azimuth <= azimuth_range[1] and
                elev_range[0] <= elev <= elev_range[1]):
            continue

        # Rotation matrix
        # The code was adopted from gluLookAt function (uses OpenGL coordinate system):
        # [1] http://stackoverflow.com/questions/5717654/glulookat-explanation
        # [2] https://www.opengl.org/wiki/GluLookAt_code
        f = -np.array(pt) # Forward direction
        f /= np.linalg.norm(f)
        u = np.array([0.0, 0.0, 1.0]) # Up direction
        s = np.cross(f, u) # Side direction
        if np.count_nonzero(s) == 0:
            # f and u are parallel, i.e. we are looking along or against Z axis
            s = np.array([1.0, 0.0, 0.0])
        s /= np.linalg.norm(s)
        u = np.cross(s, f) # Recompute up
        R = np.array([[s[0], s[1], s[2]],
                      [u[0], u[1], u[2]],
                      [-f[0], -f[1], -f[2]]])

        # Convert from OpenGL to OpenCV coordinate system
        R_yz_flip = rotation_matrix(math.pi, [1, 0, 0])[:3, :3]
        R = R_yz_flip.dot(R)

        # Translation vector
        t = -R.dot(np.array(pt).reshape((3, 1)))

        views.append({'R': R, 't': t})

    return views, pts_level




rgb = None

def generate_rgbs(rgb_path='data/images',
        label_path='data/labels',
        vis_path='data/views.ply', vis=False):
    shape = (640,480)

    model = load_ply('data/object.ply')
    model['pts'] = model['pts'] * 100.  # Scale points to match view points

    ## 9 contorls points(1 for centriod point, the rest 8 points are vertexs)
    control_points = np.array([[0,0,0],[0.2, 0.4, 0.2], [-0.2, 0.4, 0.2], [-0.2,-0.4, 0.2], [0.2,-0.4, 0.2],
                   [ 0.2,-0.4,-0.2], [ 0.2, 0.4,-0.2], [-0.2, 0.4,-0.2], [-0.2,-0.4,-0.2]])*100. # Scale points to match view points

    azimuth_range = (math.pi, 2 * math.pi)
    elev_range = (0, 0.5 * math.pi)  # (-59, 90) [deg]
    # elev_range = (0, math.pi*45/180.)
    min_n_views = 700
    views, views_level = sample_views(min_n_views, 450, azimuth_range, elev_range)
    if vis:
        save_vis(vis_path, views, views_level)

    for n in range(len(views)):
        R = views[n]['R']
        t = views[n]['t']
        mat_view = np.eye(4, dtype=np.float32)
        mat_view[:3, :3] = R
        mat_view[:3, 3] = t.squeeze()
        # print(model)
        img, xys = render_rgb(model, control_points, mat_view, shape=shape)

        cv2.imwrite(rgb_path + '/' + str(n) + '.jpg', img)
        with open(label_path + '/' + str(n) + '.txt', 'w') as f:
            for i in xys:
                f.write(str(i))
                f.write(' ')

if  __name__ == "__main__":
    import fire
    fire.Fire()