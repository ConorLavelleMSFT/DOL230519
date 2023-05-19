# Based on example found here 
# https://github.com/onnx/models/blob/main/vision/super_resolution/sub_pixel_cnn_2016/dependencies/Run_Super_Resolution_Model.ipynb
#
import onnxruntime
import matplotlib.pyplot as plt 

import SuperResolutionNet as Srn
import SuperResolutionImage as Sri

model = Srn.SuperResolutionNet(upscaleFactor=3)
srImage = Sri.SuperResolutionImage(".\Images\pexels-RobertWoeger-4991792.jpg", model.inputSize2)

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
srImage.PlotImages(True)
plt.show()