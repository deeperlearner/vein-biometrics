# python3 torch2onnx.py model/test_model_best.pth.tar --output model/palm_model_test.onnx
# python3 torch2onnx.py palm_model/test_model_best.pth.tar --output palm_model/palm_model.onnx
# python3 torch2onnx.py model_11K/test_model_best.pth.tar --output model_11K/model_11K.onnx
# python3 torch2onnx.py tongji_pretrain_11K_finetune/test_model_best.pth.tar --output tongji_pretrain_11K_finetune/model_11K.onnx
python3 torch2onnx.py model_11K/test_model_best.pth.tar --output model_11K/model_11K.onnx
# python3 torch2onnx.py tongji_pretrain_11K_finetune/test_model_best.pth.tar --output tongji_pretrain_11K_finetune/model_11K.onnx
