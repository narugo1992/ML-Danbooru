import os
import tempfile

import onnx
import onnxoptimizer
import onnxsim
import torch
from huggingface_hub import hf_hub_download

from demo import Demo
from demo import make_parser as _origin_make_parse


def make_parser():
    parser = _origin_make_parse()
    parser.add_argument('--ckpt_online', type=str, default='TResnet-D-FLq_ema_6-10000.ckpt')
    parser.add_argument('--onnx_file', type=str, default='exported.onnx')
    return parser


def export_to_onnx(model, onnx_filename, opset_version: int = 14, verbose: bool = True, no_optimize: bool = False,
                   half: bool = False):
    example_input = torch.randn((1, 3, 512, 768))
    if half:
        example_input = example_input.half()

    if torch.cuda.is_available():
        example_input = example_input.cuda()
        model = model.cuda()

    with torch.no_grad(), tempfile.TemporaryDirectory() as td:
        onnx_model_file = os.path.join(td, 'model.onnx')
        torch.onnx.export(
            model,
            example_input,
            onnx_model_file,
            verbose=verbose,
            input_names=["input"],
            output_names=["output"],

            opset_version=opset_version,
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch"},
            }
        )

        model = onnx.load(onnx_model_file)
        if not no_optimize:
            model = onnxoptimizer.optimize(model)
            model, check = onnxsim.simplify(model)
            assert check, "Simplified ONNX model could not be validated"

        output_model_dir, _ = os.path.split(onnx_filename)
        if output_model_dir:
            os.makedirs(output_model_dir, exist_ok=True)
        onnx.save(model, onnx_filename)


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if args.ckpt_online:
        args.ckpt = hf_hub_download(repo_id="7eu7d7/ML-Danbooru", filename=args.ckpt_online)

    print(args)
    demo = Demo(args)
    print(demo.model)

    export_to_onnx(demo.model, args.onnx_file, half=args.fp16)
