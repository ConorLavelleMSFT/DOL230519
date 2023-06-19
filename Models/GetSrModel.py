import urllib.request 

url = "https://github.com/onnx/models/blob/main/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
response = urllib.request.urlopen(url)
data = response.read() 

outFilePath = "super-resolution-10.onnx"
outFile = open(outFilePath, "wb")
outFile.write(data)
outFile.close()

exec("git lfs pull --include=\"super-resolution-10.onnx\" --exclude=\"\"")

url2 = "https://raw.githubusercontent.com/microsoft/onnxruntime/4324d2173b44eda7967ee2348af7c9134f8c1350/tools/python/remove_initializer_from_input.py"
response = urllib.request.urlopen(url2)
data = response.read()
text = data.decode('utf-8') 

outFilePath = "remove_initializer_from_input.py"
outFile = open(outFilePath, "w")
outFile.write(text)
outFile.close()