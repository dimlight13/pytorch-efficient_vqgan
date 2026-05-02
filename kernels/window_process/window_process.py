# --------------------------------------------------------
# Fused kernel for window process for SwinTransformer
# Copyright (c) 2022 Nvidia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch

def roll_and_window_partition_forward(x, B, H, W, C, shift_size, window_size):
    """
    Args:
        x: input tensor of shape (B*H*W, C)
        B, H, W, C: batch size, height, width, channels
        shift_size: shift size for cyclic shift
        window_size: window size for partitioning
    """
    x = x.view(B, H, W, C)
    
    # Cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    
    # Partition windows
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(-1, window_size, window_size, C)
    x = x.view(-1, window_size * window_size, C)
    
    return x

def roll_and_window_partition_backward(grad_output, B, H, W, C, shift_size, window_size):
    """Backward pass for roll_and_window_partition_forward"""
    # Reshape back to windows
    grad_output = grad_output.view(-1, window_size, window_size, C)
    grad_output = grad_output.view(B, H // window_size, W // window_size, window_size, window_size, C)
    grad_output = grad_output.permute(0, 1, 3, 2, 4, 5).contiguous()
    grad_output = grad_output.view(B, H, W, C)
    
    # Reverse cyclic shift
    if shift_size > 0:
        grad_output = torch.roll(grad_output, shifts=(shift_size, shift_size), dims=(1, 2))
    
    grad_output = grad_output.view(B * H * W, C)
    return grad_output

def window_merge_and_roll_forward(x, B, H, W, C, shift_size, window_size):
    """Merge windows and apply cyclic shift"""
    # Reshape from windowed format
    x = x.view(-1, window_size, window_size, C)
    x = x.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)
    
    # Cyclic shift
    if shift_size > 0:
        x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
    
    x = x.view(B * H * W, C)
    return x

def window_merge_and_roll_backward(grad_output, B, H, W, C, shift_size, window_size):
    """Backward pass for window_merge_and_roll_forward"""
    grad_output = grad_output.view(B, H, W, C)
    
    # Reverse cyclic shift
    if shift_size > 0:
        grad_output = torch.roll(grad_output, shifts=(-shift_size, -shift_size), dims=(1, 2))
    
    # Partition into windows
    grad_output = grad_output.view(B, H // window_size, window_size, W // window_size, window_size, C)
    grad_output = grad_output.permute(0, 1, 3, 2, 4, 5).contiguous()
    grad_output = grad_output.view(-1, window_size, window_size, C)
    grad_output = grad_output.view(-1, window_size * window_size, C)
    
    return grad_output


class WindowProcess(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = roll_and_window_partition_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W 
        ctx.C = C 
        ctx.shift_size = shift_size
        ctx.window_size = window_size
        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W 
        C = ctx.C 
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        grad_out = roll_and_window_partition_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None


class WindowProcessReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, B, H, W, C, shift_size, window_size):
        output = window_merge_and_roll_forward(input, B, H, W, C, shift_size, window_size)

        ctx.B = B
        ctx.H = H
        ctx.W = W 
        ctx.C = C 
        ctx.shift_size = shift_size
        ctx.window_size = window_size

        return output

    @staticmethod
    def backward(ctx, grad_in):
        B = ctx.B
        H = ctx.H
        W = ctx.W 
        C = ctx.C 
        shift_size = ctx.shift_size
        window_size = ctx.window_size

        #grad_out = ctx.saved_tensors[0]
        #grad_out = torch.zeros((B, H, W, C), dtype=dtype).cuda()
        grad_out = window_merge_and_roll_backward(grad_in, B, H, W, C, shift_size, window_size)
        return grad_out, None, None, None, None, None, None, None
