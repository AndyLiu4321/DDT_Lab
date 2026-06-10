import onnxruntime as ort
# sess = ort.InferenceSession("flat.onnx")
sess = ort.InferenceSession("/home/htw/ddt_lab/logs/rsl_rl/tita_flat/2026-06-08_11-17-43/exported/policy.onnx")

print("Inputs:")
for i in sess.get_inputs():
    print(i.name, i.shape)

print("\nOutputs:")
for o in sess.get_outputs():
    print(o.name, o.shape)