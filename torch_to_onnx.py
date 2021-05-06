import argparse
import torch
from craft_text_detector import CRAFT
from scatter_text_recognizer import SCATTER, AttnLabelConverter
from ocr_config import Config
import string
from ocr_utils import copyStateDict, Params


parser = argparse.ArgumentParser()
parser.add_argument(
    '-t', '--type', default='craft', type=str)
parser.add_argument(
    '--craft_model', default='craft_text_detector/weights/craft_mlt_25k.pth', type=str)
parser.add_argument(
    '--craft_output', default='craft_text_detector/weights/craft_mlt_25k.onnx', type=str)
parser.add_argument(
    '--scatter_model', default='scatter_text_recognizer/weights/scatter-case-sensitive.pth', type=str)
parser.add_argument(
    '--scatter_output', default='scatter_text_recognizer/weights/scatter-case-sensitive.onnx', type=str)
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('--craft_width', type=int, default=1280)
parser.add_argument('--craft_height', type=int, default=1024)
parser.add_argument('--scatter_width', type=int, default=1280)
parser.add_argument('--scatter_height', type=int, default=736)
parser.add_argument('-d', '--enable_dynamic_axes',
                    action="store_true", default=False)

args = parser.parse_args()
# input_size = [args.height, args.width]

if args.type == 'craft':
    dummy_input = torch.randn(
        [args.batch_size, 3, args.craft_height, args.craft_width], device='cuda')
    model = CRAFT()
    model.load_state_dict(copyStateDict(torch.load(args.craft_model)))
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
        torch.onnx.export(model, dummy_input, args.craft_output, opset_version=11,
                          verbose=True, input_names=input_names, output_names=output_names)
elif args.type == "scatter":
    # TODO
    dummy_input0 = torch.randn(
        [1, 1, 132, 100], device='cuda')
    dummy_input1 = torch.LongTensor(1, 36).fill_(0).to('cuda')
    # print(dummy_input1.shape)
    cfg = Config()
    if cfg.scatter_sensitive:
        cfg.scatter_character = string.printable[:-6]
    scatter_converter = AttnLabelConverter(cfg.scatter_character)
    if cfg.scatter_rgb:
        cfg.scatter_input_channel = 3
    scatter_params = Params(FeatureExtraction=cfg.scatter_feature_extraction, PAD=cfg.scatter_pad,
                            batch_max_length=cfg.scatter_batch_max_length, batch_size=cfg.scatter_batch_size,
                            character=cfg.scatter_character,
                            hidden_size=cfg.scatter_hidden_size, imgH=cfg.scatter_img_h, imgW=cfg.scatter_img_w,
                            input_channel=cfg.scatter_input_channel, num_fiducial=cfg.scatter_num_fiducial, num_gpu=cfg.scatter_num_gpu,
                            output_channel=cfg.scatter_output_channel,
                            rgb=cfg.scatter_rgb, saved_model=cfg.scatter_model, sensitive=cfg.scatter_sensitive,
                            workers=cfg.scatter_workers, num_class=len(scatter_converter.character))

    model = SCATTER(scatter_params)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(cfg.scatter_model), strict=False)
    model.cuda()
    model.eval()
    print(model)

    input_names = ["input0", "input1"]
    output_names = ["output_y", "output_feature"]

    if args.enable_dynamic_axes:
        dynamic_axes = {'input': [0, 2, 3], 'output_y': [0, 1, 2], 'output_feature': [0, 2, 3]}
        torch.onnx.export(model.module, dummy_input, args.scatter_output, dynamic_axes=dynamic_axes, opset_version=11,
                          verbose=True, input_names=input_names, output_names=output_names)
    else:
        torch.onnx.export(model.module, (dummy_input0, dummy_input1), args.scatter_output, opset_version=11,
                          verbose=True, input_names=input_names, output_names=output_names)
else:
    raise Exception('NOT IMPLEMENTED')
