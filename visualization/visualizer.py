# # import numpy as np
# # from vispy import app
# # from vispy import gloo

# # c = app.Canvas(keys='interactive')

# # vertex = """
# # attribute vec2 a_position;
# # void main (void){
# #     gl_Position = vec4(a_position,0.0,1.0);
# # }
# # """

# # fragment = """
# # void main() {
# #     gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
# # }
# # """

# # program = gloo.Program(vertex,fragment)

# # program['a_position'] = np.c_[np.linspace(-1.0,+1.0,1000), np.random.uniform(-0.5,+0.5,1000)].astype(np.float32)

# # @c.connect
# # def on_resize(event):
# #     gloo.set_viewport(0,0, *event.size)

# # @c.connect
# # def on_draw(event):
# #     gloo.clear((1,1,1,1))
# #     program.draw('line_strip')

# # c.show()
# # app.run();

# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # vispy: gallery 2
# # Copyright (c) Vispy Development Team. All Rights Reserved.
# # Distributed under the (new) BSD License. See LICENSE.txt for more info.

# """
# Multiple real-time digital signals with GLSL-based clipping.
# """

# from vispy import gloo
# from vispy import app
# import numpy as np
# import math

# # Number of cols and rows in the table.
# nrows = 5
# ncols = 5

# # Number of signals.
# m = nrows*ncols

# # Number of samples per signal.
# n = 1000

# # Various signal amplitudes.
# amplitudes = .1 + .2 * np.random.rand(m, 1).astype(np.float32)

# print(amplitudes.shape)

# # Generate the signals as a (m, n) array.
# y = amplitudes * np.random.randn(m, n).astype(np.float32)
# print(y.shape)

# # Color of each vertex (TODO: make it more efficient by using a GLSL-based
# # color map and the index).
# color = np.repeat(np.random.uniform(size=(m, 3), low=.5, high=.9),
#                   n, axis=0).astype(np.float32)
# print(color.shape)

# # Signal 2D index of each vertex (row and col) and x-index (sample index
# # within each signal).
# index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n),
#               np.repeat(np.tile(np.arange(nrows), ncols), n),
#               np.tile(np.arange(n), m)].astype(np.float32)
# print(index.shape)

# VERT_SHADER = """
# #version 120

# // y coordinate of the position.
# attribute float a_position;

# // row, col, and time index.
# attribute vec3 a_index;
# varying vec3 v_index;

# // 2D scaling factor (zooming).
# uniform vec2 u_scale;

# // Size of the table.
# uniform vec2 u_size;

# // Number of samples per signal.
# uniform float u_n;

# // Color.
# attribute vec3 a_color;
# varying vec4 v_color;

# // Varying variables used for clipping in the fragment shader.
# varying vec2 v_position;
# varying vec4 v_ab;

# void main() {
#     float nrows = u_size.x;
#     float ncols = u_size.y;

#     // Compute the x coordinate from the time index.
#     float x = -1 + 2*a_index.z / (u_n-1);
#     vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);

#     // Find the affine transformation for the subplots.
#     vec2 a = vec2(1./ncols, 1./nrows)*.9;
#     vec2 b = vec2(-1 + 2*(a_index.x+.5) / ncols,
#                   -1 + 2*(a_index.y+.5) / nrows);
#     // Apply the static subplot transformation + scaling.
#     gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);

#     v_color = vec4(a_color, 1.);
#     v_index = a_index;

#     // For clipping test in the fragment shader.
#     v_position = gl_Position.xy;
#     v_ab = vec4(a, b);
# }
# """

# FRAG_SHADER = """
# #version 120

# varying vec4 v_color;
# varying vec3 v_index;

# varying vec2 v_position;
# varying vec4 v_ab;

# void main() {
#     gl_FragColor = v_color;

#     // Discard the fragments between the signals (emulate glMultiDrawArrays).
#     if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
#         discard;

#     // Clipping test.
#     vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
#     if ((test.x > 1) || (test.y > 1))
#         discard;
# }
# """


# class Canvas(app.Canvas):
#     def __init__(self):
#         app.Canvas.__init__(self, title='Use your wheel to zoom!',
#                             keys='interactive')
#         self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
#         self.program['a_position'] = y.reshape(-1, 1)
#         self.program['a_color'] = color
#         self.program['a_index'] = index
#         self.program['u_scale'] = (1., 1.)
#         self.program['u_size'] = (nrows, ncols)
#         self.program['u_n'] = n

#         gloo.set_viewport(0, 0, *self.physical_size)

#         self._timer = app.Timer('auto', connect=self.on_timer, start=True)

#         gloo.set_state(clear_color='black', blend=True,
#                        blend_func=('src_alpha', 'one_minus_src_alpha'))

#         self.show()

#     def on_resize(self, event):
#         gloo.set_viewport(0, 0, *event.physical_size)

#     def on_mouse_wheel(self, event):
#         dx = np.sign(event.delta[1]) * .05
#         scale_x, scale_y = self.program['u_scale']
#         scale_x_new, scale_y_new = (scale_x * math.exp(2.5*dx),
#                                     scale_y * math.exp(0.0*dx))
#         self.program['u_scale'] = (max(1, scale_x_new), max(1, scale_y_new))
#         self.update()

#     def on_timer(self, event):
#         """Add some data at the end of each signal (real-time signals)."""
#         k = 10
#         y[:, :-k] = y[:, k:]
#         y[:, -k:] = amplitudes * np.random.randn(m, k)

#         self.program['a_position'].set_data(y.ravel().astype(np.float32))
#         self.update()

#     def on_draw(self, event):
#         gloo.clear()
#         self.program.draw('line_strip')

# if __name__ == '__main__':
#     c = Canvas()
#     app.run()

import numpy as np
from vispy import gloo, app
from vispy.gloo import set_viewport, set_state, clear

vert = """
#version 120

// Uniforms
// ------------------------------------
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_antialias;
uniform float u_size;

// Attributes
// ------------------------------------
attribute vec3  a_position;
attribute vec4  a_fg_color;
attribute vec4  a_bg_color;
attribute float a_linewidth;
attribute float a_size;

// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;

void main (void) {
    v_size = a_size * u_size;
    v_linewidth = a_linewidth;
    v_antialias = u_antialias;
    v_fg_color  = a_fg_color;
    v_bg_color  = a_bg_color;
    gl_Position = u_projection * u_view * u_model *
        vec4(a_position*u_size,1.0);
    gl_PointSize = v_size + 2*(v_linewidth + 1.5*v_antialias);
}
"""

frag = """
#version 120

// Constants
// ------------------------------------

// Varyings
// ------------------------------------
varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_size;
varying float v_linewidth;
varying float v_antialias;

// Functions
// ------------------------------------
float marker(vec2 P, float size);


// Main
// ------------------------------------
void main()
{
    float size = v_size +2*(v_linewidth + 1.5*v_antialias);
    float t = v_linewidth/2.0-v_antialias;

    // The marker function needs to be linked with this shader
    float r = marker(gl_PointCoord, size);

    float d = abs(r) - t;
    if( r > (v_linewidth/2.0+v_antialias))
    {
        discard;
    }
    else if( d < 0.0 )
    {
       gl_FragColor = v_fg_color;
    }
    else
    {
        float alpha = d/v_antialias;
        alpha = exp(-alpha*alpha);
        if (r > 0)
            gl_FragColor = vec4(v_fg_color.rgb, alpha*v_fg_color.a);
        else
            gl_FragColor = mix(v_bg_color, v_fg_color, alpha);
    }
}

float marker(vec2 P, float size)
{
    float r = length((P.xy - vec2(0.5,0.5))*size);
    r -= v_size/2;
    return r;
}
"""

vs = """
attribute vec3 a_position;
attribute vec4 a_fg_color;
attribute vec4 a_bg_color;
attribute float a_size;
attribute float a_linewidth;

void main(){
    gl_Position = vec4(a_position, 1.);
}
"""

fs = """
void main(){
    gl_FragColor = vec4(0., 0., 0., 1.);
}
"""


class Canvas(app.Canvas):

    def __init__(self, **kwargs):
        # Initialize the canvas for real
        app.Canvas.__init__(self, keys='interactive', size=(512, 512),
                            **kwargs)
        ps = self.pixel_scale
        self.position = 50, 50

        n = 100
        ne = 100
        data = np.zeros(n, dtype=[('a_position', np.float32, 3),
                                  ('a_fg_color', np.float32, 4),
                                  ('a_bg_color', np.float32, 4),
                                  ('a_size', np.float32, 1),
                                  ('a_linewidth', np.float32, 1),
                                  ])
        edges = np.random.randint(size=(ne, 2), low=0,
                                  high=n).astype(np.uint32)
        data['a_position'] = np.hstack((.25 * np.random.randn(n, 2),
                                       np.zeros((n, 1))))
        data['a_fg_color'] = 0, 0, 0, 1
        color = np.random.uniform(0.5, 1., (n, 3))
        data['a_bg_color'] = np.hstack((color, np.ones((n, 1))))
        data['a_size'] = np.random.randint(size=n, low=8*ps, high=20*ps)
        data['a_linewidth'] = 1.*ps
        u_antialias = 1

        self.vbo = gloo.VertexBuffer(data)
        self.index = gloo.IndexBuffer(edges)
        self.view = np.eye(4, dtype=np.float32)
        self.model = np.eye(4, dtype=np.float32)
        self.projection = np.eye(4, dtype=np.float32)

        self.program = gloo.Program(vert, frag)
        self.program.bind(self.vbo)
        self.program['u_size'] = 1
        self.program['u_antialias'] = u_antialias
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.program['u_projection'] = self.projection

        set_viewport(0, 0, *self.physical_size)

        self.program_e = gloo.Program(vs, fs)
        self.program_e.bind(self.vbo)

        set_state(clear_color='white', depth_test=False, blend=True,
                  blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def on_resize(self, event):
        set_viewport(0, 0, *event.physical_size)

    def on_draw(self, event):
        clear(color=True, depth=True)
        self.program_e.draw('lines', self.index)
        self.program.draw('points')

if __name__ == '__main__':
    c = Canvas(title="Graph")
    app.run()