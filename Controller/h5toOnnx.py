from tensorflow.keras.models import load_model
import keras2onnx

model = load_model('lane.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, 'lane_v1.onnx')
