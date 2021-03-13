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


class CTM(nn.Module):

    def __init__(self, args, in_channels=64):
        super().__init__()
        self.args = args

        blocks = args.ctm_blocks
        out_channels = args.ctm_out_channels if args.ctm_out_channels > 0 else in_channels
        diff_channels = in_channels - out_channels

        if args.ctm_block_type == 'ConvBlock':
            make_layer = _make_conv_layer
        else:
            raise ValueError('Invalid CTM block type')

        ### Concentrator
        in_concentrator = in_channels
        out_concentrator = round(in_channels - (diff_channels / 2)) if in_channels != out_channels else in_channels
        if args.ctm_m_type == 'fused':
            in_concentrator *= args.shot
        self.concentrator = make_layer(blocks, in_concentrator, out_concentrator, args.ctm_reduce_dims)

        ### Projector
        in_projector = out_concentrator
        if args.ctm_m_type == 'fused':
            in_projector *= args.way
        self.projector = make_layer(blocks, in_projector, out_channels, args.ctm_reduce_dims)

        ### Reshaper
        self.reshaper = make_layer(blocks, in_channels, out_channels, args.ctm_reduce_dims, 5)

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

        