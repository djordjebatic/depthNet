import torch
import torch.nn.functional as F

def multiscale_loss(output, disp_true, weights = None):
    ''' Weighted multiscale loss.
    '''
    pr1, pr2, pr3, pr4, pr5, pr6 = output

    # predict flow upsampling
    out_size = pr1.shape[-2:]
    out1 = pr1
    out2 = F.interpolate(pr2, out_size, mode='bilinear')
    out3 = F.interpolate(pr3, out_size, mode='bilinear')
    out4 = F.interpolate(pr4, out_size, mode='bilinear')
    out5 = F.interpolate(pr5, out_size, mode='bilinear')
    out6 = F.interpolate(pr6, out_size, mode='bilinear')

    # squeeze
    out1 = torch.squeeze(out1, 1)
    out2 = torch.squeeze(out2, 1)
    out3 = torch.squeeze(out3, 1)
    out4 = torch.squeeze(out4, 1)
    out5 = torch.squeeze(out5, 1)
    out6 = torch.squeeze(out6, 1)

    # weights
    if weights is None:
        weights = [0.0025, 0.005, 0.01, 0.02, 0.08, 0.32]

    # weighted multiscale loss
    outs = (out6, out5, out4, out3, out2, out1)
    loss = 0
    for w, o in zip(weights, outs):
        loss_delta = w * F.smooth_l1_loss(o, disp_true, size_average=True)
        loss += loss_delta

    return loss