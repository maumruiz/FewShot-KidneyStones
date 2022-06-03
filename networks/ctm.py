import torch
import torch.nn as nn

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        # nn.MaxPool2d(2)
    )

def _make_conv_layer(blocks, in_channels, out_channels, reduce_dims = False, reduce_kernel=3):
    layers = []
    for i in range(blocks):
        layers.append(conv_block(in_channels, out_channels))
        in_channels = out_channels

    if reduce_dims:
        layers.append(nn.MaxPool2d(reduce_kernel, stride=1))
    return nn.Sequential(*layers)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def _make_res_layer(blocks, inplanes, planes, reduce_dims = False, reduce_kernel=3, stride=1):
    planes = round(planes / 4)
    downsample = None
    if stride != 1 or inplanes != planes * Bottleneck.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * Bottleneck.expansion,
                        kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * Bottleneck.expansion),
        )
    layers = []
    layers.append(Bottleneck(inplanes, planes, stride, downsample))
    inplanes = planes * Bottleneck.expansion
    for i in range(1, blocks):
        layers.append(Bottleneck(inplanes, planes))
    
    if reduce_dims:
        layers.append(nn.MaxPool2d(reduce_kernel, stride=1))

    return nn.Sequential(*layers)

class CTM(nn.Module):

    def __init__(self, args, in_channels=64):
        super().__init__()
        self.args = args

        blocks = args.ctm_blocks
        out_channels = args.ctm_out_channels if args.ctm_out_channels > 0 else in_channels
        diff_channels = in_channels - out_channels

        if args.ctm_block_type == 'ConvBlock':
            make_layer = _make_conv_layer
        elif args.ctm_block_type == 'ResBlock':
            make_layer = _make_res_layer
        else:
            raise ValueError('Invalid CTM block type')

        ### Concentrator
        in_concentrator = in_channels
        out_concentrator = round(in_channels - (diff_channels / 2)) if in_channels != out_channels else in_channels
        if args.ctm_m_type == 'fused':
            in_concentrator *= args.shot

        if not args.ctm_split_blocks:
            self.concentrator = make_layer(blocks, in_concentrator, out_concentrator, args.ctm_reduce_dims)
        else:
            diff_blocks = blocks - 3
            self.concentrator = nn.Sequential(
                make_layer(3, in_concentrator, round(out_concentrator*2)),
                make_layer(diff_blocks, round(out_concentrator*2), round(out_concentrator), args.ctm_reduce_dims)
            )

        ### Projector
        in_projector = out_concentrator
        if args.ctm_m_type == 'fused':
            in_projector *= args.way

        if not args.ctm_split_blocks:
            self.projector = make_layer(blocks, in_projector, out_channels, args.ctm_reduce_dims)
        else:
            diff_blocks = blocks - 3
            self.projector = nn.Sequential(
                make_layer(3, in_projector, round(out_channels*2)),
                make_layer(diff_blocks, round(out_channels*2), round(out_channels), args.ctm_reduce_dims)
            )

        ### Reshaper
        if not args.ctm_split_blocks:
            self.reshaper = make_layer(blocks, in_channels, out_channels, args.ctm_reduce_dims, 5)
        else:
            diff_blocks = blocks - 3
            self.reshaper = nn.Sequential(
                make_layer(3, in_channels, round(out_channels*2)),
                make_layer(diff_blocks, round(out_channels*2), round(out_channels), args.ctm_reduce_dims, 5)
            )

    def forward(self, supp_fts, query_fts):
        # Support fts: WaxSh, C(64), H'(5), W'(5)
        # Query fts: Q, C(64), H'(5), W'(5)

        fts_size = supp_fts.shape[-2:]
        in_channels = supp_fts.size(1)


        ### CONCENTRATOR
        in_concentrator = supp_fts
        if self.args.ctm_m_type == 'fused':
            # Reshape to have (n_way, n_shot*in_channels, H, W)
            in_concentrator = supp_fts.view(self.args.way, -1, *fts_size)

        out_concentrator = self.concentrator(in_concentrator)


        ### PROJECTOR
        fts_size = out_concentrator.shape[-2:]
        if self.args.ctm_m_type == 'fused':
            # Reshape to have (1, n_way*in_channels, d2, d2)
            in_projector = out_concentrator.view(1, -1, *fts_size)
        elif self.args.ctm_m_type == 'avg':
            # Reshape to have (n_way, concentrator_channels, d2, d2)
            in_projector = torch.mean(
                out_concentrator.view(self.args.way, self.args.shot, out_concentrator.size(1), *fts_size), 
                dim=1, keepdim=False
                )

        out_projector = self.projector(in_projector)

        if self.args.ctm_m_type == 'avg':
            # Reshape to have (1, out_channels, d3, d3)
            out_projector = torch.mean(out_projector, dim=0, keepdim=True)

        
        ### ENHANCED FEATURES
        out_supp_fts = self.reshaper(supp_fts)
        out_supp_fts = torch.matmul(out_supp_fts, out_projector)

        out_query_fts = self.reshaper(query_fts)
        out_query_fts = torch.matmul(out_query_fts, out_projector)

        return out_supp_fts, out_query_fts

        