"""Learn OpenGL with Qt for Python.

This program follows: https://learnopengl.com/Getting-started/Hello-Triangle

Compare it against the C++ implementation:
https://learnopengl.com/code_viewer_gh.php?code=src/1.getting_started/2.1.hello_triangle/hello_triangle.cpp

Other useful resources include:
- https://paroj.github.io/gltut/Basics/Tut01%20Following%20the%20Data.html
- https://web.archive.org/web/20160408075002/http://www.labri.fr/perso/nrougier/teaching/opengl/
- https://www.khronos.org/opengl/wiki/Main_Page
- http://pyopengl.sourceforge.net/documentation/manual-3.0/index.html#GL

and, of course, the PySide2 documentation.

"""

import sys

import numpy as np
import ctypes

# Used to access OpenGL enums
from OpenGL import GL as pygl

from PySide2 import QtWidgets, QtCore, QtGui
import shiboken2


FLOAT_SIZE = ctypes.sizeof(ctypes.c_float)


class GLWidget(QtWidgets.QOpenGLWidget):

    def __init__(self, parent=None):
        # OpenGL is a client-server architecture. The graphics engine,
        # which runs on the GPU, is the "server".  Buffers
        # (e.g. vertex buffer object) and GPU procedures
        # (i.e. shaders) exist server-side.  The applications using
        # the OpenGL API, such as this application, are the "clients".
        # Broadly speaking, objects returned by the API are handles to
        # GPU objects; the OpenGL API is merely an interface.  Qt
        # provides the drawing window (this widget) and sets the
        # context (i.e. the OpenGL GPU state-machine).
        super().__init__(parent)

        # This GLWidget will draw a triangle.  A triangle consists of
        # three points in 2D.  Each point is called a vertex.  OpenGL
        # processes data represented as 3D or 4D vectors.  When 2D
        # vertices are represented as 3D vectors, the z-coordinate
        # should be set to 0.  OpenGL only processes 'normalized
        # device coordinates'; values between -1.0 and 1.0.  Any data
        # outside this range will not be displayed.  Data may be
        # normalized prior to loading, as done here, or afterwards.
        self.vertex_data = np.array(
            [-0.5, -0.5, 0.0,   # x, y, z
              0.5, -0.5, 0.0,   # x, y, z
              0.0,  0.5, 0.0],  # x, y, z
            dtype=ctypes.c_float
        )

        # Qt allows OpenGL objects to be instantiated prior to
        # initialization.  Several objects are created here
        # (instantiated, actually) but it's not until
        # QtWidgets.QOpenGLWidget.initializeGL() is called that they
        # can be initialized.  The deferred initialization requires
        # developers to manually initialize these objects (via
        # corresponding create() method calls).

        # The Vertex Buffer Object (VBO) is used to manage memory in
        # the GPU.  It allocates space and specifies the usage pattern
        # for the data (default is GL_STATIC_DRAW). As the name
        # implies, the VBO is an OpenGL object; it represents a subset
        # of OpenGL's state.  Note that the object provided here (by
        # the API) is simply a handle.  The actual buffer will exist
        # on the GPU.  It's not until
        # QtWidgets.QOpenGLWidget.initializeGL() is called, and the
        # vbo is created, that the object is actually initialized on
        # the GPU.
        #
        # For descriptions of usage patterns, see
        # https://doc-snapshots.qt.io/qtforpython-5.15/PySide2/QtGui/QOpenGLBuffer.html?highlight=qopenglbuffer#PySide2.QtGui.PySide2.QtGui.QOpenGLBuffer.UsagePattern
        self.vbo = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.VertexBuffer)

        # Data is passed to the GPU and processed through a pipeline,
        # or sequence of steps.  Each step is represented by a
        # program, or procedure, called a "shader" which converts some
        # input into an output ready for the next step.  Shaders are
        # written in a domain specific language called OpenGL Shading
        # Language (GLSL) that looks like C.  Although there are many
        # possible shaders (i.e. steps in the processing chain), only
        # two are required: the vertex shader and the fragment shader.
        # The vertex shader appears early in the pipline and handles
        # coordinate transformations of the data passed to the GPU
        # through the VBO.  The fragment shader, sometimes called the
        # pixel shader, appears at the end of the pipeline and
        # calculates the color output associated with each piece of
        # data.  Shaders must be compiled and linked within the GPU.
        # Qt requires this be handled in QOpenGLWidget.initializeGL().
        #
        # More information about GLSL and links to resources can be
        # found on the OpenGL wiki:
        # https://www.khronos.org/opengl/wiki/Core_Language_(GLSL)
        #
        # More information about the rendering pipeline can be read
        # here:
        # https://www.khronos.org/opengl/wiki/Rendering_Pipeline_Overview

        # The OpenGL Shading Language requires certain information to
        # be presented early in a shader's compilation.  In a
        # command-line-based compiler, these would be command-line
        # options.  GLSL's compilation model instead requires these
        # options to be part of the language.  Shader compiler options
        # are given in the first line of the string defining the
        # shader.  This line must begin with a GLSL version
        # specification, unless the default v1.10 is desired
        # (unlikely).  The version is followed by a profile name,
        # 'core' (default) or 'compatibility'.  Historically, OpenGL
        # used the "Fixed Function Pipeline".  This was deprecated in
        # OpenGL v3.0 and removed in OpenGL v3.1.  OpenGL v3.2
        # reintroduced the old functionality due to continued vendor
        # implementation.  Declaring a profile reconciles the two
        # approaches.
        #
        # Vertex shader inputs are called 'attributes' and are defined
        # with an 'in' statement.  Here an attribute called 'aPos' of
        # type vec3 is created.  Shader attributes are positional and
        # each is assigned an index using the 'layout (location = i)'
        # syntax where 0 <= i <= 16.  The upper limit is fixed in
        # hardware.  The attribute 'aPos' occupies index 0.
        # Attributes are always referenced by their index outside of
        # the shader.
        #
        # Just like in C, the entry point for a shader is the 'main'
        # function.  Here, the inputs are converted from a vec3 to a
        # vec4 and stored in the standard output variable
        # 'gl_Position'.
        self.vertex_shader_code = """
        #version 330 core
        layout (location = 0) in vec3 aPos;

        void main()
        {
            gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        }
        """

        # Fragment shaders require a single output: a vector of size 4
        # giving the color in RGBA.
        self.fragment_shader_code = """
        #version 330 core
        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
        }
        """

        # Core OpenGL, as was specified in the shaders, requires a
        # Vertex Array Object (VAO).  A VA0 is used to switch between
        # contexts and stores buffer and attribute states.  Because of
        # this, a VAO should be bound before a VBO.
        self.vao = QtGui.QOpenGLVertexArrayObject()

    def getGlInfo(self):
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
            """.format(
            pygl.glGetString(pygl.GL_VENDOR),
            pygl.glGetString(pygl.GL_RENDERER),
            pygl.glGetString(pygl.GL_VERSION),
            pygl.glGetString(pygl.GL_SHADING_LANGUAGE_VERSION)
        )
        return info

    def initializeGL(self):
        """Required when subclassing QtWidgets.QOpenGLWidget.

        Used to set up any required OpenGL resources and state
        (e.g. vertex and fragment shaders).

        QtWidgets.QOpenGLWidget.makeCurrent() is called automatically,
        making an OpenGL context available.

        Avoid issuing draw calls at this stage, as the framebuffer is
        not yet available.

        """

        # OpenGL is now available.  Grab and display its info.
        print(self.getGlInfo(), flush=True)

        # There are several ways to interact with OpenGL:
        #
        #   1. Qt objects
        #   2. PyOpenGL
        #
        # PyOpenGL provides direct access to enums and functions.  The
        # Qt documentation states,
        #
        #   When making OpenGL function calls, it is strongly
        #   recommended to avoid calling the functions
        #   directly. Instead, prefer using QOpenGLFunctions. This way
        #   the application will work correctly in all Qt build
        #   configurations, including the ones that perform dynamic
        #   OpenGL implementation loading which means applications are
        #   not directly linking to an GL implementation and thus
        #   direct function calls are not feasible.
        #
        # See: https://doc-snapshots.qt.io/qtforpython-5.15/PySide2/QtWidgets/QOpenGLWidget.html?highlight=qopenglwidget#opengl-function-calls-headers-and-qopenglfunctions
        # See: https://doc-snapshots.qt.io/qtforpython-5.15/PySide2/QtGui/QOpenGLFunctions.html#more
        #
        # The simplest way to access OpenGL functions is through the
        # context instance.  The only way, AFAICT, to access OpenGL
        # enums is through PyOpenGL.
        gl_funcs = self.context().functions()

        # The minimal OpenGL application requires a vertex shader and
        # a fragment shader.  Shaders are GLSL code which has been
        # compiled and linked within the GPU.  Qt handles compiling
        # with a QtGui.QOpenGLShader object and linking with a
        # QtGui.QOpenGLShaderProgram.

        # In order for the shader to be compiled, an OpenGL context
        # must exist.  The QOpenGLWidget.initializeGL() method
        # automatically creates a context for the window containing
        # this widget. The QtGui.QOpenGLWidget.context().surface() is
        # a PySide2.QtGui.QOffscreenSurface object.  This is a primary
        # difference between the QOpenGLWidget and the legacy
        # QGLWidget.

        # Here, the shaders are defined only within the scope of
        # initialization.  Remember that the shaders actually exist on
        # the GPU and that the objects here are merely the interface.
        # Therefore, these objects do not need an extended lifetime
        # beyond their immediate use (i.e. no need for 'self').
        vertex_shader = QtGui.QOpenGLShader(QtGui.QOpenGLShader.Vertex)
        is_compiled = vertex_shader.compileSourceCode(self.vertex_shader_code)
        if not is_compiled:
            raise ValueError("Vertex shader did not compile:\n {0}".format(self.vertex_shader_code))

        fragment_shader = QtGui.QOpenGLShader(QtGui.QOpenGLShader.Fragment)
        is_compiled = fragment_shader.compileSourceCode(self.fragment_shader_code)
        if not is_compiled:
            raise ValueError("Fragment shader did not compile:\n {0}".format(self.fragment_shader_code))

        # The vertex and fragment shaders are linked into a single
        # program.  This program is called whenever rendering occurs.
        # Rendering is the process of converting higher dimension data
        # representations into the 2D images displayed on-screen
        # (i.e. the shader pipeline).  Rendering happens whenever the
        # object is redrawn (e.g. window resize), so the program
        # should be stored in an attribute for future reference.
        self.program = QtGui.QOpenGLShaderProgram()
        self.program.addShader(vertex_shader)
        self.program.addShader(fragment_shader)
        self.program.bindAttributeLocation("aPos", 0)

        self.program.link()
        print("Program linked: ", self.program.isLinked(), flush=True)

        is_program_bound = self.program.bind()
        print('Program bound: ', is_program_bound, flush=True)

        # Now that a context exists, the VAO and VBO can be created.

        # VAO - bind before VBO to capture state!
        self.vao.create()
        print('VAO created: ', self.vao.isCreated(), flush=True)

        self.vao_binder = QtGui.QOpenGLVertexArrayObject.Binder(self.vao)
        print('VAO bound: ', self.vao_binder, flush=True)

        # VBO
        self.vbo.create()
        print('VBO created: ', self.vbo.isCreated(), flush=True)

        is_vbo_bound = self.vbo.bind()
        print('VBO bound: ', is_vbo_bound, flush=True)

        # Once the VBO has been created and bound to the current
        # context, copy data over to the GPU.  Note that the size of
        # the data must be defined.  This step is akin to calling
        # 'glBufferData'.  The main difference is that the usage
        # pattern was defined when the VBO was instantiated.
        self.vbo.allocate(self.vertex_data.tobytes(), FLOAT_SIZE * self.vertex_data.size)

        # The vertex shader needs to know how to parse the data.  This
        # is done with 'glVertexAttribPointer' which defines an array
        # of generic vertex attribute data. It does this with whatever
        # buffer is bound to the current context. The vertex shader is
        # executed for each vertex that given to the rendering
        # pipeline.  This information is captured by the VAO.
        gl_funcs.glVertexAttribPointer(0,                    # index of vertex attribute
                                       3,                    # size of each vertex; 3 for vec3
                                       int(pygl.GL_FLOAT),   # type of each element in the array
                                       int(pygl.GL_FALSE),   # should the shader normalize the data?
                                       3 * FLOAT_SIZE,       # space between consecutive attributes (stride); each attribute is vec3
                                       shiboken2.VoidPtr(0)  # offset of where data begins
                                       )

        # Enable the (shader) vertex attribute at index 0.  This must
        # be called prior to rendering.  Which attributes are
        # enabled/disabled is tracked by the VAO.
        gl_funcs.glEnableVertexAttribArray(0)

        # The VAO records the (un)binding of the VBO.  Be sure to
        # unbind the VAO *before* unbinding the VBO (unless the
        # release is to be recorded).
        self.vao_binder.release()
        self.vbo.release()
        self.program.release()

    def paintGL(self):
        """Render the OpenGL scene.

        Gets called whenever the widget needs to be updated.

        """

        # clean up what was drawn
        funcs = self.context().functions()
        funcs.glClear(pygl.GL_COLOR_BUFFER_BIT)

        # Binding here is redundant since there is only a single
        # context
        self.vao_binder.rebind()

        # Bind which shaders to apply
        self.program.bind()

        # Render primitives (e.g. triangles) from array data
        funcs.glDrawArrays(pygl.GL_TRIANGLES, # what kind of primitives to render
                           0,                 # starting index in the enabled arrays
                           3)                 # number of indices to read; done in 3s because GL_TRIANGLES

        self.program.release()
        self.vao_binder.release()


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, self.on_exit)

        self.init_widgets()
        self.init_layout()

    def init_widgets(self):
        self.exit_action = QtWidgets.QAction()
        self.exit_action.setText("&Quit")
        self.exit_action.triggered.connect(self.on_exit)

        self.menu_bar = QtWidgets.QMenuBar()
        self.menu_file = self.menu_bar.addMenu('&File')
        self.menu_file.addAction(self.exit_action)
        self.setMenuBar(self.menu_bar)

        self.x_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.x_slider.setRange(0, 360 * 16)
        self.x_slider.setSingleStep(16)
        self.x_slider.setPageStep(15 * 16)
        self.x_slider.setTickInterval(15 * 16)
        self.x_slider.setTickPosition(QtWidgets.QSlider.TicksRight)

        self.y_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.y_slider.setRange(0, 360 * 16)
        self.y_slider.setSingleStep(16)
        self.y_slider.setPageStep(15 * 16)
        self.y_slider.setTickInterval(15 * 16)
        self.y_slider.setTickPosition(QtWidgets.QSlider.TicksRight)

        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Vertical)
        self.z_slider.setRange(0, 360 * 16)
        self.z_slider.setSingleStep(16)
        self.z_slider.setPageStep(15 * 16)
        self.z_slider.setTickInterval(15 * 16)
        self.z_slider.setTickPosition(QtWidgets.QSlider.TicksRight)

        self.gl_widget = GLWidget()

    def init_layout(self):
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.gl_widget)

        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def on_exit(self):
        self.close()



if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
