#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from model import DETR
from train import get_parser
from nets.backbone import build_backbone
from nets.transformer import build_transformer

if __name__ == "__main__":
    input_shape     = [600, 600]
    num_classes     = 21
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = get_parser()
    args = parser.parse_args()

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    num_classes = 91
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        device=device,
    )

    model = model.to(device)
    
    # summary(model, (3, input_shape[0], input_shape[1]))
    # if summary errors, use `print(model) to show the architecture`
    print(model)

    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
