"""
LRGAN_FUNCTIONS
Extra functions for LRGAN

Stefan Wong 2019
"""

import torch
from torch.autograd import Function

# TODO : how to import the spatial transformer network?


class STNFunction(Function):
    def forward(self,
                canvas,
                fgimg:torch.Tensor,
                fggrid:torch.Tensor,
                fgmask:torch.Tensor) -> torch.Tensor:
        self.canvas = canvas
        self.fgimg  = fgimg
        self.fggrid = fggrid
        self.fgmask = fgmask

        output = torch.zeros(
            canvas.size()[0],
            canvas.size()[1],
            canvas.size()[2],
            canvas.size()[3]
        )

        if not canvas.is_cuda:
            raise ValueError('CUDA required for STNFunction.forward()')

        stnm.BillinearSampler_BHWD_updateOutput_cuda(canvas, fgimg, fggrid, fgmask, output)

        return output

    def backward(self, dz:torch.Tensor) -> tuple:
        d_canvas = torch.zeros(self.canvas.size())
        d_fgimg  = torch.zeros(self.fgimg.size())
        d_fggrid = torch.zeros(self.fggrid.size())
        d_fgmask = torch.zeros(self.fgmask.size())

        if not dz.is_cuda:
            raise ValueError('CUDA required for STNFunction.backward()')

        dz = dz.contiguous()
        d_canvas = d_canvas.contiguous()
        d_fgimg  = d_fgimg.contiguous()
        d_fggrid  = d_fggrid.contiguous()
        d_fgmask  = d_fgmask.contiguous()

        stnm.BillinearSampler_BHWD_updateGradInput_cuda(
            self.canvas,
            self.fgimg,
            self.fggrid,
            self.fgmask,
            d_canvas,
            d_fgimg,
            d_fggrid,
            d_fgmask,
            dz
        )

        return (d_canvas, d_fgimg, d_fggrid, d_fgmask)
