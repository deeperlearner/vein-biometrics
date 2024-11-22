import os
import argparse

import numpy as np
import onnx
import torch

from models import MNASNet_Modified


def convert_onnx(net, path_module, output, opset=11, simplify=False):
    assert isinstance(net, torch.nn.Module)
    img = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.int32)
    img = img.astype(float)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()

    checkpoint = torch.load(path_module)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    torch.onnx.export(net, img, output, input_names=['data'],output_names = ['output'],
                        dynamic_axes={'data' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}},
                        keep_initializers_as_inputs=False, verbose=False, opset_version=opset)
    model = onnx.load(output)
    #graph = model.graph
    #graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(model, output)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--output', type=str, default=None, help='output onnx path')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file)
    # model_name = os.path.basename(os.path.dirname(input_file)).lower()
    # params = model_name.split("_")
    # if len(params) >= 3 and params[1] in ('arcface', 'cosface'):
    #     if args.network is None:
    #         args.network = params[2]
    print(args)
    backbone_onnx = MNASNet_Modified(embedding_size=512, class_size=100, only_embeddings=True, pretrained=True)
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), "model.onnx")
    convert_onnx(backbone_onnx, input_file, args.output, simplify=args.simplify)
