# Based on example found here 
# https://github.com/onnx/models/blob/main/vision/super_resolution/sub_pixel_cnn_2016/dependencies/Run_Super_Resolution_Model.ipynb
#
from __future__ import absolute_import
import onnxruntime
from pyglet import gl
import pygame 
from imgui.integrations.pygame import PygameRenderer
import OpenGL.GL as gl
import imgui
from imgui.integrations.cocos2d import ImguiLayer
import matplotlib.pyplot as plt 
import sys

import SuperResolutionNet as Srn
import SuperResolutionImage as Sri

def getGLTexture(image):
    textureData = list(image.getdata())
    width = image.size[0]
    height = image.size[1]

    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB,
                    gl.GL_UNSIGNED_BYTE, textureData)

    return texture, width, height


def imguiFrame(srImage):
    imgui.new_frame()
    srImage.ShowImagesImgui()

def mainPygame(srImage):
    pygame.init()
    size = 1200, 900
    pygame.display.set_mode(size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
    imgui.create_context()
    renderer = PygameRenderer()
    io = imgui.get_io()
    io.display_size = size

    srImage.GenerateGLTextures()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            renderer.process_event(event)
        renderer.process_inputs()
        
        imguiFrame(srImage)
        gl.glClearColor(1, 0, 1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        renderer.render(imgui.get_draw_data())
        
        pygame.display.flip()

model = Srn.SuperResolutionNet(upscaleFactor=3) 
srImage = Sri.SuperResolutionImage(".\Images\pexels-GuntherZ-5561853.jpg", model.inputSize2)

# Run Model
# Original Model can be found here, "-corrected" means run through subsequent remove_init... script
#   https://github.com/onnx/models/tree/main/vision/super_resolution/sub_pixel_cnn_2016
#   https://github.com/microsoft/onnxruntime/blob/main/tools/python/remove_initializer_from_input.py
modelFile = ".\Models\super-resolution-10-corrected.onnx"
ortSession = onnxruntime.InferenceSession(modelFile, providers=['CPUExecutionProvider'])
ortInputs = {ortSession.get_inputs()[0].name: srImage.processedInput} 
ortOutputs = ortSession.run(None, ortInputs)

# Postprocess and draw
srImage.FinalizeImage(ortOutputs[0])
#srImage.PlotImages()
#srImage.PlotImages(True)
#plt.show()

mainPygame(srImage)