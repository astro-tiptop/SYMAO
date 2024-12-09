from symao.turbolence import *

turbolenceFormulas = createTurbolenceFormulary()

import numpy as np
import cupy as cp
import cupyx.scipy.linalg as scipyx_linalg

def seeing_to_r0(seeing, lamda=500.E-9):
    return 0.98*lamda/(seeing*np.pi/(180.*3600.))

from scipy.special import gamma, kv

class PhaseScreen(object):
    def __init__(self, mx_size, pixel_scale, r0, L0, l0, xp=np, random_seed=None, stencil_size_factor=1):
        self.requested_mx_size = mx_size
        print(self.requested_mx_size)
        self.mx_size = 2 ** (int( np.ceil(np.log2(mx_size)))) + 1
        print(self.mx_size)
        self.pixel_scale = pixel_scale
        self.r0 = r0
        self.L0 = L0
        self.l0 = l0
        self.xp = xp
        self.stencil_size_factor = stencil_size_factor
        self.stencil_size = stencil_size_factor * self.mx_size        
        if random_seed is not None:
            self.xp.random.seed(random_seed)
        #self.set_stencil_coords_basic()
        self.set_stencil_coords()
        self.setup()

    def phase_covariance(self, r, r0, L0):
        # Make sure everything is a float to avoid nasty surprises in division!
        r = self.xp.asnumpy(r)
        r0 = float(r0)
        L0 = float(L0)
        # Get rid of any zeros
        r += 1e-40
        A = (L0 / r0) ** (5. / 3)
        B1 = (2 ** (-5. / 6)) * gamma(11. / 6) / (self.xp.pi ** (8. / 3))
        B2 = ((24. / 5) * gamma(6. / 5)) ** (5. / 6)
        C = (((2 * self.xp.pi * r) / L0) ** (5. / 6)) * kv(5. / 6, (2 * self.xp.pi * r) / L0)
        cov = A * B1 * B2 * C
        cov = self.xp.asarray(cov)
        return cov

    def set_stencil_coords_basic(self):
        self.stencil = self.xp.zeros((self.stencil_size, self.stencil_size))
        self.stencil[:2,:] = 1
        self.stencil_coords = self.xp.array(self.xp.where(self.stencil==1)).T
        self.stencil_positions = self.stencil_coords * self.pixel_scale
        self.n_stencils = self.stencil_coords.shape[0]


    def set_stencil_coords(self):
        self.stencil = np.zeros((self.stencil_size, self.stencil_size))
        self.stencilF = np.zeros((self.stencil_size, self.stencil_size))
        max_n = int( np.floor(np.log2(self.stencil_size)))
        # Fill the  head of stencil (basiaccaly all of it for us)
        for n in range(0, max_n + 1):
            col = int((2 ** (n - 1)) + 1)
            n_points = (2 ** (max_n - n)) + 1
            coords = np.round(np.linspace(0, self.stencil_size - 1, n_points)).astype('int32')
            self.stencil[col - 1][coords] = 1
            self.stencilF[self.stencil_size - col][coords] = 1
        # Fill the tail of stencil
        for n in range(1, self.stencil_size_factor + 1):
            col = n * self.mx_size - 1
            self.stencil[col, self.stencil_size // 2] = 1
            self.stencilF[self.stencil_size-col-1, self.stencil_size // 2] = 1

        self.stencil = self.xp.asarray(self.stencil)
        self.stencilF = self.xp.asarray(self.stencilF)
        self.stencil_coords = []
        self.stencil_coords.append(self.xp.array(self.xp.where(self.stencil == 1)).T)
        self.stencil_coords.append(self.xp.array(self.xp.where(self.stencilF == 1)).T)
        self.stencil_positions = []
        self.stencil_positions.append(self.stencil_coords[0] * self.pixel_scale)
        self.stencil_positions.append(self.stencil_coords[1] * self.pixel_scale)        
        self.n_stencils = self.stencil_coords[0].shape[0]

    def AB_from_positions(self, positions):
        seperations = self.xp.zeros((len(positions), len(positions)))
        px, py = positions[:,0], positions[:,1]
        delta_x_gridA, delta_x_gridB = self.xp.meshgrid(px, px)
        delta_y_gridA, delta_y_gridB = self.xp.meshgrid(py, py)
        delta_x_grid = delta_x_gridA - delta_x_gridB
        delta_y_grid = delta_y_gridA - delta_y_gridB
        seperations = self.xp.sqrt(delta_x_grid ** 2 + delta_y_grid ** 2)
        # make cov_mats
        self.cov_mat = self.phase_covariance(seperations, self.r0, self.L0)
        print('self.cov_mat', self.cov_mat.shape)
        self.cov_mat_zz = self.cov_mat[:self.n_stencils, :self.n_stencils]
        self.cov_mat_xx = self.cov_mat[self.n_stencils:, self.n_stencils:]
        self.cov_mat_zx = self.cov_mat[:self.n_stencils, self.n_stencils:]
        self.cov_mat_xz = self.cov_mat[self.n_stencils:, :self.n_stencils]
        # print('self.cov_mat_zz', self.cov_mat_zz.shape)
        # print('self.cov_mat_xx', self.cov_mat_xx.shape)
        # print('self.cov_mat_zx', self.cov_mat_zx.shape)
        # print('self.cov_mat_xz', self.cov_mat_xz.shape)
        # make A
        # Cholesky solve can fail - so do brute force inversion
        cf = scipyx_linalg.lu_factor(self.cov_mat_zz)
        inv_cov_zz = scipyx_linalg.lu_solve(cf, self.xp.identity(self.cov_mat_zz.shape[0]))
        A_mat = self.cov_mat_xz.dot(inv_cov_zz)
        # make B
        # Can make initial BBt matrix first
        BBt = self.cov_mat_xx - A_mat.dot(self.cov_mat_zx)
        # Then do SVD to get B matrix
        u, W, ut = self.xp.linalg.svd(BBt)
        L_mat = self.xp.zeros((self.stencil_size, self.stencil_size))
        self.xp.fill_diagonal(L_mat, self.xp.sqrt(W))
        # Now use sqrt(eigenvalues) to get B matrix
        B_mat = u.dot(L_mat)
        return A_mat, B_mat
    
    def setup(self):
        # set X coords
        self.new_col_coords1 = self.xp.zeros((self.stencil_size, 2))
        self.new_col_coords1[:, 0] = -1
        self.new_col_coords1[:, 1] = self.xp.arange(self.stencil_size)
        self.new_col_positions1 = self.new_col_coords1 * self.pixel_scale
        #self.new_col_coords2 = self.xp.zeros((self.stencil_size, 2))
        #self.new_col_coords2[:, 0] = self.stencil_size
        #self.new_col_coords2[:, 1] = self.xp.arange(self.stencil_size)
        #self.new_col_positions2 = self.new_col_coords2 * self.pixel_scale
        # calc separations
        positions1 = self.xp.concatenate((self.stencil_positions[0], self.new_col_positions1), axis=0)
        #positions2 = self.xp.concatenate((self.stencil_positions[1], self.new_col_positions2), axis=0)        
        self.A_mat, self.B_mat = [], []
        A_mat, B_mat = self.AB_from_positions(positions1)
        self.A_mat.append(A_mat)
        self.B_mat.append(B_mat)
        # A_mat, B_mat = self.AB_from_positions(positions2)
        self.A_mat.append(cp.fliplr(cp.flipud(A_mat)))
        self.B_mat.append(B_mat)
        # print(cp.max(self.B_mat[0]-self.B_mat[1]))
        # print(cp.max(self.A_mat[0]-cp.fliplr(cp.flipud(self.A_mat[1]))))
        # make initial screen
        self._scrn = cp.asarray(ft_phase_screen( turbolenceFormulas, self.r0, self.stencil_size, self.pixel_scale, self.L0, self.l0 ))
        #self._scrn = cp.zeros((self.stencil_size, self.stencil_size))
        print(self._scrn.shape)  

    def get_new_line(self, row, after):
        random_data = self.xp.random.normal(size=self.stencil_size) / ((2*self.xp.pi)**2)
        if row:
            stencil_data = self.xp.asarray(self._scrn[self.stencil_coords[after][:, 1], self.stencil_coords[after][:, 0]])
        else:
            stencil_data = self.xp.asarray(self._scrn[self.stencil_coords[after][:, 0], self.stencil_coords[after][:, 1]])            
        new_line = self.A_mat[after].dot(stencil_data) + self.B_mat[after].dot(random_data)        
        return new_line

    def add_line(self, row, after):
        new_line = self.get_new_line(row, after)
        if row:
            new_line = new_line[:,self.xp.newaxis]
            if after:
                self._scrn = self.xp.concatenate((self._scrn, new_line), axis=row)[:self.stencil_size, 1:]
            else:
                self._scrn = self.xp.concatenate((new_line, self._scrn), axis=row)[:self.stencil_size, :self.stencil_size]
        else:
            new_line = new_line[self.xp.newaxis, :]
            if after:
                self._scrn = self.xp.concatenate((self._scrn, new_line), axis=row)[1:, :self.stencil_size]
            else:
                self._scrn = self.xp.concatenate((new_line, self._scrn), axis=row)[:self.stencil_size, :self.stencil_size]

    @property
    def scrn(self):
        return cp.asnumpy(self._scrn[:self.requested_mx_size, :self.requested_mx_size])

    @property
    def scrnRaw(self):
        return self._scrn[:self.requested_mx_size, :self.requested_mx_size]

import sys
import time

from cuda import cudart

import numpy as np
import cupy as cp

import pyrr
import glfw

from OpenGL.GL import *  # noqa F403
import OpenGL.GL.shaders

OpenGL.ERROR_CHECKING = False

def format_cudart_err(err):
    return (
        f"{cudart.cudaGetErrorName(err)[1].decode('utf-8')}({int(err)}): "
        f"{cudart.cudaGetErrorString(err)[1].decode('utf-8')}"
    )


def check_cudart_err(args):
    if isinstance(args, tuple):
        assert len(args) >= 1
        err = args[0]
        if len(args) == 1:
            ret = None
        elif len(args) == 2:
            ret = args[1]
        else:
            ret = args[1:]
    else:
        err = args
        ret = None

    assert isinstance(err, cudart.cudaError_t), type(err)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(format_cudart_err(err))

    return ret


class CudaOpenGLMappedBuffer:
    def __init__(self, gl_buffer, flags=0):
        self._gl_buffer = int(gl_buffer)
        self._flags = int(flags)

        self._graphics_ressource = None
        self._cuda_buffer = None

        self.register()

    @property
    def gl_buffer(self):
        return self._gl_buffer

    @property
    def cuda_buffer(self):
        assert self.mapped
        return self._cuda_buffer

    @property
    def graphics_ressource(self):
        assert self.registered
        return self._graphics_ressource

    @property
    def registered(self):
        return self._graphics_ressource is not None

    @property
    def mapped(self):
        return self._cuda_buffer is not None

    def __enter__(self):
        return self.map()

    def __exit__(self, exc_type, exc_value, trace):
        self.unmap()
        return False

    def __del__(self):
        self.unregister()

    def register(self):
        if self.registered:
            return self._graphics_ressource
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsGLRegisterBuffer(self._gl_buffer, self._flags)
        )
        return self._graphics_ressource

    def unregister(self):
        if not self.registered:
            return self
        self.unmap()
        self._graphics_ressource = check_cudart_err(
            cudart.cudaGraphicsUnregisterResource(self._graphics_ressource)
        )
        return self

    def map(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot map an unregistered buffer.")
        if self.mapped:
            return self._cuda_buffer

        check_cudart_err(
            cudart.cudaGraphicsMapResources(1, self._graphics_ressource, stream)
        )

        ptr, size = check_cudart_err(
            cudart.cudaGraphicsResourceGetMappedPointer(self._graphics_ressource)
        )

        self._cuda_buffer = cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(ptr, size, self), 0
        )

        return self._cuda_buffer

    def unmap(self, stream=None):
        if not self.registered:
            raise RuntimeError("Cannot unmap an unregistered buffer.")
        if not self.mapped:
            return self

        self._cuda_buffer = check_cudart_err(
            cudart.cudaGraphicsUnmapResources(1, self._graphics_ressource, stream)
        )

        return self


class CudaOpenGLMappedTexture(CudaOpenGLMappedBuffer):
    def __init__(self, dtype, shape, gl_buffer, flags=0, strides=None, order='C'):
        super().__init__(gl_buffer, flags)
        self._dtype = dtype
        self._shape = shape
        self._strides = strides
        self._order = order

    @property
    def cuda_array(self):
        assert self.mapped
        return cp.ndarray(
            shape=self._shape,
            dtype=self._dtype,
            strides=self._strides,
            order=self._order,
            memptr=self._cuda_buffer,
        )

    def map(self, *args, **kwargs):
        super().map(*args, **kwargs)
        return self.cuda_array


class CudaOpenGLMappedArray(CudaOpenGLMappedBuffer):
    def __init__(self, dtype, shape, gl_buffer, flags=0, strides=None, order='C'):
        super().__init__(gl_buffer, flags)
        self._dtype = dtype
        self._shape = shape
        self._strides = strides
        self._order = order

    @property
    def cuda_array(self):
        assert self.mapped
        return cp.ndarray(
            shape=self._shape,
            dtype=self._dtype,
            strides=self._strides,
            order=self._order,
            memptr=self._cuda_buffer,
        )

    def map(self, *args, **kwargs):
        super().map(*args, **kwargs)
        return self.cuda_array


VERTEX_SHADER = """
#version 120

in vec3 aVertex;

uniform mat4 transform;

void main() {
    gl_Position = transform * vec4(aVertex, 1.0f);
    gl_TexCoord[0] = gl_MultiTexCoord0;
}
"""


FRAGMENT_SHADER = """
#version 120

uniform sampler2D textureObj;

float colormap_red(float x) {
    if (x < 0.7) {
        return 4.0 * x - 1.5;
    } else {
        return -4.0 * x + 4.5;
    }
}

float colormap_green(float x) {
    if (x < 0.5) {
        return 4.0 * x - 0.5;
    } else {
        return -4.0 * x + 3.5;
    }
}

float colormap_blue(float x) {
    if (x < 0.3) {
       return 4.0 * x + 0.5;
    } else {
       return -4.0 * x + 2.5;
    }
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = clamp(colormap_blue(x), 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

void main()
{   
    float cc;
    cc = (texture2D(textureObj, gl_TexCoord[0].st).r+1.5)/3.0;
    gl_FragColor = colormap(cc);
}

"""

def main():

    NN = 2048
    pixel_scale = 8.2/float(NN)
    print('Creating phase screen..')
    scrn = PhaseScreen(NN, pixel_scale, seeing_to_r0(0.8), 20, 0.005, cp)
    print('... phase screen created')
        
    if not glfw.init():
        return
    title = "CuPy Cuda/OpenGL interop example"
    window = glfw.create_window(1024, 1024, title, None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.swap_interval(0)

    shader = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER),
    )
    positionLoc = glGetAttribLocation(shader, "position")
    transformLoc = glGetUniformLocation(shader, "transform")

    fps = 0
    nframes = 0
    last_time = glfw.get_time()

    glUseProgram(shader)
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_CULL_FACE)
    glEnable(GL_TEXTURE_2D)

    bufferID = glGenBuffers(1)
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferID)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, NN*NN*4, scrn.scrn, GL_DYNAMIC_DRAW);
    ftype = np.float32
    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    texture_buffer = CudaOpenGLMappedTexture(ftype, (NN, NN), bufferID, flags)

    textureID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, textureID)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, NN, NN, 0, GL_RED, GL_FLOAT, None)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    # could be inside the loop if actually applying a transform
    tt = pyrr.Matrix44.from_translation([-1,-1,0]) 
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, tt)

    one_quad_dl = glGenLists(1)
    glNewList(one_quad_dl, GL_COMPILE)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex(0, 0, 0)
    glTexCoord2f(0, 1)
    glVertex(2, 0, 0)
    glTexCoord2f(1, 1)
    glVertex(2, 2, 0)
    glTexCoord2f(1, 0)
    glVertex(0, 2, 0)
    glEnd()
    glEndList()

    wind_angle=0.0
    rx = 0
    ry = 0
    while not glfw.window_should_close(window):
        t = glfw.get_time()
        dt = t - last_time
        if dt >= 1.0:
            fps = nframes / dt
            last_time = t
            nframes = 0

        width, height = glfw.get_window_size(window)
        glViewport(0, 0, width, height)
        
        wind_angle += 0.8 * 0.0174533 # one degree 
        winds_x = np.sin(wind_angle)*4.0 + rx
        winds_y = np.cos(wind_angle)*4.0 + ry

        fx, ix = np.modf(winds_x)
        fy, iy = np.modf(winds_y)
        
        if np.abs(ix)>0.0:
            for i in range(int(np.abs(ix))):
                if winds_x>0:
                    scrn.add_line(0,1)
                else:
                    scrn.add_line(0,0)
                rx = fx
        else:
            rx += fx

        if np.abs(iy)>0.0:
            for i in range(int(np.abs(iy))):
                if winds_y>0:
                    scrn.add_line(1,1)
                else:
                    scrn.add_line(1,0)
                ry = fy
        else:
            ry += fy
                    
        time.sleep(0.005)
    
        with texture_buffer as TT:
            TT[:,:] = scrn.scrnRaw
            glTexSubImage2D( GL_TEXTURE_2D, 0, 0, 0, NN, NN, GL_RED, GL_FLOAT, None)

        glActiveTexture(GL_TEXTURE0)
        glMatrixMode(GL_MODELVIEW)
        glCallList(one_quad_dl)

        glfw.swap_buffers(window)
        glfw.poll_events()
        glfw.set_window_title(window, f"{title} ({fps:.1f} fps)")
        nframes += 1

    texture_buffer.unregister()
    glfw.terminate()


if __name__ == "__main__":
    sys.exit(main())