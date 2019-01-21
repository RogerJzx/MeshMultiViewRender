## Mesh Multi-View Render tools

This Project uses ply format mesh to produce multi-views 2D  rgb  images

#### Dependencies

glumpy is python openGL. http://glumpy.github.io/

#### About OpenGl Coordinates Transform

- OpenGL坐标变换 https://www.zybuluo.com/ltlovezh/note/911669?utm_source=wechat_session&utm_medium=social&utm_oi=698607419638435840
- OpenGL Projection Matrix http://www.songho.ca/opengl/gl_projectionmatrix.html
- Camera Instrisic Matrix to Projection Matrix  https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL

#### Usage
- python main.py generate_rgbs --rgb_path='data/images'  --label_path='data/labels'  --vis_path="You view mesh path"  --vis=True
    It can generate view points on a sphere, rgb images and 9 controls points.
    if vis=True: it saves a ply mesh in your path folder to debug and visualization, you can watch it on meshlab

