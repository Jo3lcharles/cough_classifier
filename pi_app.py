import tflite_runtime.interpreter as tflite
import numpy as np

# Mock function (replace with real audio later)
def predict_cough():
    # Simulate MFCC input
    mfcc = np.random.rand(13).astype('float32')
    
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path="models/cough_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    
    # Predict
    interpreter.set_tensor(input_details[0]['index'], [mfcc])
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    
    return "Abnormal" if output > 0.5 else "Normal"

print("Prediction:", predict_cough())