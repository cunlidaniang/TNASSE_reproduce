import numpy as np
import math

import torch
import torch.nn as nn

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv=nn.Conv2d(in_channels=int(in_channels), 
                            out_channels=int(out_channels),
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False)
        self.batch_norm=nn.BatchNorm2d(num_features=int(out_channels))
        self.relu=nn.ReLU()
        self.to(self.device)

    def forward(self, x):
        x=self.conv(x)
        x=self.batch_norm(x)
        x=self.relu(x)
        return x

class Conv3x3BnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv3x3BnRelu, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv3x3 = ConvBnRelu(in_channels, out_channels, 3, 1, 1)
        self.to(self.device)

    def forward(self, x):
        x = self.conv3x3(x)
        return x

class Conv1x1BnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1BnRelu, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1x1 = ConvBnRelu(in_channels, out_channels, 1, 1, 0)
        self.to(self.device)

    def forward(self, x):
        x = self.conv1x1(x)
        return x

class MaxPool3x3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MaxPool3x3, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)
        self.to(self.device)

    def forward(self, x):
        x = self.maxpool(x)
        return x

OP_MAP = {
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3
}

# 通过Cell构建Network
class Network(nn.Module):
    def __init__(self, spec, device, out_channels=16, stacknum=3, modules_per_stack=3, label_num=1, searchspace=[]):
        super(Network, self).__init__()

        self.layers = nn.ModuleList([])
        self.device=device
        self.to(self.device)

        in_channels = 3

        stem_conv = ConvBnRelu(in_channels, out_channels, 3, 1, 1)
        self.layers.append(stem_conv)
        in_channels = out_channels
        for stack_num in range(stacknum):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(downsample)
                out_channels *= 2

            for module_num in range(modules_per_stack):
                cell = Cell(spec, in_channels, out_channels)
                self.layers.append(cell)
                in_channels = out_channels

        self.classifier = nn.Linear(out_channels, label_num)

        num_edge = np.shape(spec.matrix)[0]
        self.arch_parameters = nn.Parameter( 1e-3*torch.randn(num_edge, len(searchspace)))

        self._initialize_weights()

    def forward(self, x, get_ints=True):
        ints = []
        for _, layer in enumerate(self.layers):
            x=x.to(self.device)
            x = layer(x)
            ints.append(x)
        x=x.to(self.device)
        out = torch.mean(x, (2, 3))
        ints.append(out)
        out=out.to(self.device)
        out = self.classifier(out)
        if get_ints:
            return out, ints[-1]
        else:
            return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                pass
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                pass
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                pass

    # def get_weights(self):
    #     xlist = []
    #     for m in self.modules():
    #         xlist.append(m.parameters())
    #     return xlist

    # def get_alphas(self):
    #     return [self.arch_parameters]

# 根据spec构造Cell
class Cell(nn.Module):
    def __init__(self, spec, in_channels, out_channels):
        super(Cell, self).__init__()

        self.spec = spec
        self.num_vertices = np.shape(self.spec.matrix)[0]
        self.vertex_channels = ComputeVertexChannels(in_channels, out_channels, self.spec.matrix)

        self.vertex_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices-1):
            op = OP_MAP[spec.ops[t]](self.vertex_channels[t], self.vertex_channels[t])
            self.vertex_op.append(op)

        self.input_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices):
            if self.spec.matrix[0, t]:
                self.input_op.append(Projection(in_channels, self.vertex_channels[t]))
            else:
                self.input_op.append(None)

    def forward(self, x):
        tensors = [x]
        out_concat = []
        for t in range(1, self.num_vertices-1):
            fan_in = [Truncate(tensors[src], self.vertex_channels[t]) for src in range(1, t) if self.spec.matrix[src, t]]
            fan_in_inds = [src for src in range(1, t) if self.spec.matrix[src, t]]
            if self.spec.matrix[0, t]:
                fan_in.append(self.input_op[t](x))
                fan_in_inds = [0] + fan_in_inds
            vertex_input = sum(fan_in)
            vertex_output = self.vertex_op[t](vertex_input)
            tensors.append(vertex_output)
            if self.spec.matrix[t, self.num_vertices-1]:
                out_concat.append(tensors[t])

        if not out_concat:
            assert self.spec.matrix[0, self.num_vertices-1]
            outputs = self.input_op[self.num_vertices-1](tensors[0])
        else:
            if len(out_concat) == 1:
                outputs = out_concat[0]
            else:
                outputs = torch.cat(out_concat, 1)
            if self.spec.matrix[0, self.num_vertices-1]:
                outputs += self.input_op[self.num_vertices-1](tensors[0])
        return outputs

def Projection(in_channels, out_channels):
    return ConvBnRelu(in_channels=in_channels,out_channels=out_channels)

#将inputs提取对应通道数
def Truncate(inputs, channels):
    input_channels = inputs.size()[1]
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs 
    else:
        assert input_channels - channels == 1
        return inputs[:, :int(channels), :, :]
    
# 计算每个点的通道数
def ComputeVertexChannels(in_channels, out_channels, matrix):
    num_vertices = np.shape(matrix)[0]
    vertex_channels = [0] * num_vertices
    vertex_channels[0] = in_channels
    vertex_channels[num_vertices - 1] = out_channels
    if num_vertices == 2:
        return vertex_channels
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = out_channels // in_degree[num_vertices - 1]
    correction = out_channels % in_degree[num_vertices - 1]
    for v in range(1, num_vertices - 1):
      if matrix[v, num_vertices - 1]:
          vertex_channels[v] = interior_channels
          if correction:
              vertex_channels[v] += 1
              correction -= 1
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == out_channels or num_vertices == 2
    return vertex_channels