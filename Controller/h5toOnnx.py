from tensorflow.keras.models import load_model
import keras2onnx

model = load_model('model-047.h5')
onnx_model = keras2onnx.convert_keras(model,model.name)
keras2onnx.save_model(onnx_model,'sign_new.onnx')
