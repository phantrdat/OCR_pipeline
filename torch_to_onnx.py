import argparse
import torch
from craft_text_detector import CRAFT
from ocr_utils import copyStateDict

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='craft_text_detector/weights/craft_mlt_25k.pth', type=str)
parser.add_argument(
    '-o', '--output', default='craft_text_detector/weights/craft_mlt_25k_da.onnx', type=str)
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('--width', type=int, default=1280)
parser.add_argument('--height', type=int, default=736)
parser.add_argument('-d', '--enable_dynamic_axes',
                    action="store_true", default=False)

args = parser.parse_args()
input_size = [args.height, args.width]
dummy_input = torch.randn(
    [args.batch_size, 3, args.height, args.width], device='cuda')
model = CRAFT()
model.load_state_dict(copyStateDict(torch.load(args.model)))
model.cuda()
model.eval()
# print(model)

input_names = ["input"]
output_names = ["output_y", "output_feature"]

if args.enable_dynamic_axes:
    dynamic_axes = {'input': [0, 2, 3], 'output_y': [0, 1, 2], 'output_feature': [0, 2, 3]}
    torch.onnx.export(model, dummy_input, args.output, dynamic_axes=dynamic_axes, opset_version=11,
                      verbose=True, input_names=input_names, output_names=output_names)
else:
    torch.onnx.export(model, dummy_input, args.output, opset_version=11,
                      verbose=True, input_names=input_names, output_names=output_names)
