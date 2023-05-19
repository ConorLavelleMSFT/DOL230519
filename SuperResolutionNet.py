import torch
import torch.nn as nn 
import torch.nn.init as init 
import torch.utils.model_zoo as model_zoo

# From Super Resolution Model Sample in Onnx Repo 
# https://github.com/onnx/models/blob/main/vision/super_resolution/sub_pixel_cnn_2016/dependencies/Run_Super_Resolution_Model.ipynb
class SuperResolutionNet(nn.Module):
    def __init__(self, upscaleFactor, inplace=False):
        super(SuperResolutionNet, self).__init__()
        self.inputSize = 224
        self.inputSize2 = [self.inputSize, self.inputSize]
        # Rectified Linear Unit Function
        self.relu = nn.ReLU(inplace=inplace)
        # 2d convolution (InChannels, OutChannels, Kernel, Stride, Padding)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscaleFactor ** 2, (3, 3), (1, 1), (1, 1))
        # helper for rearranging elements in a tensor based on an upscale factor 
        self.pixel_shuffle = nn.PixelShuffle(upscaleFactor)

        self._initializeWeights()
        self._loadPretrainedWeights()
        self._setToInferenceMode()

    # override from nn.Module, defines computation performed at every call 
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initializeWeights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

    def _loadPretrainedWeights(self):
        modelUrl = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
        mapLocation = lambda storage, loc: storage
        self.load_state_dict(model_zoo.load_url(modelUrl, map_location=mapLocation))

    def _setToInferenceMode(self):
        self.eval()
        x = torch.randn(1, 1, self.inputSize, self.inputSize, requires_grad=True)
        self.eval()
