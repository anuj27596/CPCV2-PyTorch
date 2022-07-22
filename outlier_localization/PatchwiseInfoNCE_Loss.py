import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class InfoNCE_Loss(nn.Module):
    """Performs predictions and InfoNCE Loss

    Modified From:
    https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax/vision/models/InfoNCE_Loss.py
    https://github.com/loeweX/Greedy_InfoMax/blob/master/LICENSE

    Args:
        pred_steps (int): number of steps into the future to perform predictions
        neg_samples (int): number of negative samples to be used for contrastive loss
        in_channels (int): number of channels of input tensors (size of encoding vector from encoder network and autoregressive network)
    """
    
    def __init__(self, pred_steps, neg_samples, in_channels):
        super().__init__()
        
        self.pred_steps = pred_steps
        self.neg_samples = neg_samples

        self.W_k = nn.ModuleList(
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
            for _ in range(self.pred_steps)
        )

        self.contrast_loss = ExpNLLLoss()

    def forward(self, z, c, skip_step=1):
        batch_size = z.shape[0]
        total_loss = 0
        cur_device = z.device#get_device()

        patchwise_loss = torch.zeros((batch_size, *z.shape[-2:]), device=cur_device)
        patchwise_count = torch.zeros(z.shape[-2:], device=cur_device)

        # For each element in c, contrast with elements below
        for k in range(1, self.pred_steps + 1):
            ### compute log f(c_t, x_{t+k}) = z^T_{t+k} * W_k * c_t
            
            ### compute z^T_{t+k} * W_k:
            ztwk = (
                self.W_k[k - 1]
                .forward(z[:, :, (k + skip_step) :, :])  # Bx, C , H , W
                .permute(2, 3, 0, 1)  # H, W, Bx, C
                .contiguous()
            )  # y, x, b, c

            ztwk_shuf = ztwk.view(
                ztwk.shape[0] * ztwk.shape[1] * ztwk.shape[2], ztwk.shape[3]
            )  # y * x * batch, c
            rand_index = torch.randint(
                ztwk_shuf.shape[0],  # y *  x * batch
                (ztwk_shuf.shape[0] * self.neg_samples, 1),
                dtype=torch.long,
                device=cur_device,
            )
            # Sample more
            rand_index = rand_index.repeat(1, ztwk_shuf.shape[1])

            ztwk_shuf = torch.gather(
                ztwk_shuf, dim=0, index=rand_index, out=None
            )  # y * x * b * n, c

            ztwk_shuf = ztwk_shuf.view(
                ztwk.shape[0],
                ztwk.shape[1],
                ztwk.shape[2],
                self.neg_samples,
                ztwk.shape[3],
            ).permute(
                0, 1, 2, 4, 3
            )  # y, x, b, c, n

            ### Compute  x_W1 * c_t:
            context = (
                c[:, :, : -(k + skip_step), :].permute(2, 3, 0, 1).unsqueeze(-2)
            )  # y, x, b, 1, c

            log_fk_main = torch.matmul(context, ztwk.unsqueeze(-1)).squeeze(
                -2
            )  # y, x, b, 1

            log_fk_shuf = torch.matmul(context, ztwk_shuf).squeeze(-2)  # y, x, b, n

            log_fk = torch.cat((log_fk_main, log_fk_shuf), 3)  # y, x, b, 1+n
            log_fk = log_fk.permute(2, 3, 0, 1)  # b, 1+n, y, x

            log_fk = torch.softmax(log_fk, dim=1)

            true_f = torch.zeros(
                (batch_size, log_fk.shape[-2], log_fk.shape[-1]),
                dtype=torch.long,
                device=cur_device,
            )  # b, y, x

            total_loss += self.contrast_loss(input=log_fk, target=true_f)


            stu_loss = -torch.log(log_fk[:, 0, :, :] + 1e-11)

            patchwise_loss[:, (k + skip_step) :, :] += stu_loss
            patchwise_loss[:, :-(k + skip_step), :] += stu_loss
            patchwise_count[(k + skip_step) :, :] += torch.ones_like(log_fk[0, 0, :, :])
            patchwise_count[:-(k + skip_step), :] += torch.ones_like(log_fk[0, 0, :, :])

            # patchwise_loss[:, (k + skip_step) :, :] += stu_loss * patchwise_loss.size(2) / log_fk.size(2) / self.pred_steps / 2
            # patchwise_loss[:, :-(k + skip_step), :] += stu_loss * patchwise_loss.size(2) / log_fk.size(2) / self.pred_steps / 2

        total_loss /= self.pred_steps

        # patchwise_loss /= patchwise_count


        return total_loss, patchwise_loss, patchwise_count


class ExpNLLLoss(_WeightedLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(ExpNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        x = torch.log(input + 1e-11)
        return F.nll_loss(x, target, weight=self.weight, ignore_index=self.ignore_index,
                          reduction=self.reduction)



if __name__ == '__main__':
    
    batch_size = 16
    grid_size = 5
    encoding_size = 64

    pred_directions = 4

    pred_loss = [InfoNCE_Loss(pred_steps=3, neg_samples=7, in_channels=encoding_size) for _ in range(pred_directions)]

    # z = torch.randn((batch_size, encoding_size, grid_size, grid_size))
    # c = torch.randn((batch_size, encoding_size, grid_size, grid_size))

    # with torch.no_grad():
    #     l, p = pred_loss(z, c)

    # print('z', z.size())
    # print('c', c.size())
    # print('l', l.size())
    # print('p', p.size())

    # print('l', l)
    # print('mu(p)', p.nanmean())

    with torch.no_grad():
        encodings = torch.randn((batch_size, encoding_size, grid_size, grid_size))

        tloss = 0
        pwloss = 0
        pwcnt = 0
        for i in range(pred_directions):
            # rotate encoding 90 degrees clockwise
            if i > 0:
                encodings = encodings.transpose(2,3).flip(3)
                pwloss = pwloss.transpose(1,2).flip(2)
                pwcnt = pwcnt.transpose(0,1).flip(1)

            # Find all context vectors
            contexts = torch.randn((batch_size, encoding_size, grid_size, grid_size)) # (batch_size, encoding_size, grid_size, grid_size)

            # Find contrastive pwloss
            l, p, c = pred_loss[i](encodings, contexts)

            tloss += l
            pwloss += p
            pwcnt += c

        tloss /= pred_directions
        pwloss /= pwcnt
        # pwloss /= pred_directions
