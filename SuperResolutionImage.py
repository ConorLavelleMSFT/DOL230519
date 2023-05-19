import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

class SuperResolutionImage():
    # Preprocess source image and store intermediates
    def __init__(self, sourceImagePath, modelInputSize):
        self.finalImage = None
        self.bicubicResize = None
        # Open image and scale to match model input
        self.sourceImage = Image.open(sourceImagePath)
        self.scaledImage = self.sourceImage.resize(modelInputSize)
        # Convert to YCbCr then get components 
        # Y: Greyscale, Cb: Change in Blue, Cr: Change in Red
        scaledInputYcbcr = self.scaledImage.convert('YCbCr')
        self.scaledInputY, self.scaledInputCb, self.scaledInputCr = scaledInputYcbcr.split()
        scaledInputYArray = np.asarray(self.scaledInputY)
        # Insert additional dimension for channels and batch
        scaledInput4D = np.expand_dims(np.expand_dims(scaledInputYArray, axis=0), axis=0)
        # Normalize 
        self.processedInput = scaledInput4D.astype(np.float32) / 255.0

    def FinalizeImage(self, onnxOut):
        if (self.finalImage == None):
            onnxUnnormalized = np.uint8((onnxOut[0] * 255.0).clip(0, 255)[0])
            outputY = Image.fromarray(onnxUnnormalized, mode='L')
            # SR is only acting on greyscale, merge back in the 
            # Original sample notes this follows post-processing step from PyTorch implementation
            self.finalImage = Image.merge(
                "YCbCr", [
                    outputY,
                    self.scaledInputCb.resize(outputY.size, Image.BICUBIC),
                    self.scaledInputCr.resize(outputY.size, Image.BICUBIC),
                ]).convert("RGB")

    # Sets up for draw but does not call show, caller must call show
    def PlotImages(self, gridLayout=False):   
        if (gridLayout):
            r = self.finalImage.size[1] / self.sourceImage.size[1]
            fig, subPlots = plt.subplots(
                2, 2,  # Row, Col
                figsize=(16,9), 
                gridspec_kw={'height_ratios' : [1.0, r]})
            flattenedSubplots = [
            subPlots[0][0], subPlots[0][1], 
            subPlots[1][0], subPlots[1][1]]
        else:
            r1 = self.sourceImage.size[0] / self.scaledImage.size[0]
            r2 = self.finalImage.size[0] / self.scaledImage.size[0]
            fig, subPlots = plt.subplots(
                1, 4,  # Row, Col
                figsize=(16,9), 
                gridspec_kw={'width_ratios' : [r1, 1.0, r2, r2]})
            flattenedSubplots = subPlots

        ax = fig.add_subplot(flattenedSubplots[0])
        ax.set_title('Source')
        plt.imshow(self.sourceImage, interpolation="none")

        ax = fig.add_subplot(flattenedSubplots[1])
        ax.set_title('Scaled Input')
        plt.imshow(self.scaledImage, interpolation="none")

        ax = fig.add_subplot(flattenedSubplots[2])
        ax.set_title('Bicubic')
        if (self.bicubicResize == None):
            self.bicubicResize = self.scaledImage.resize(self.finalImage.size, Image.BICUBIC)
        plt.imshow(self.bicubicResize, interpolation="none")

        ax = fig.add_subplot(flattenedSubplots[3])
        ax.set_title('Onnx SR')
        plt.imshow(self.finalImage, interpolation="none")

