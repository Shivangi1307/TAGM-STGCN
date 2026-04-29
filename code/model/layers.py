import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ================= ALIGN =================
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(c_in, c_out, kernel_size=(1,1))

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.align_conv(x)
        elif self.c_in < self.c_out:
            B,_,T,N = x.shape
            pad = torch.zeros([B,self.c_out-self.c_in,T,N], device=x.device)
            return torch.cat([x,pad],dim=1)
        return x

# ================= TEMPORAL =================
class TemporalConvLayer(nn.Module):
    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super().__init__()
        self.Kt = Kt
        self.align = Align(c_in,c_out)
        self.act_func = act_func


        if act_func in ['glu','gtu']:
            self.conv = nn.Conv2d(c_in,2*c_out,(Kt,1))
        else:
            self.conv = nn.Conv2d(c_in,c_out,(Kt,1))
        self.relu = nn.ReLU()

    def forward(self,x):
        x_in = self.align(x)[:,:,self.Kt-1:,:]
        x_conv = self.conv(x)


        if self.act_func in ['glu','gtu']:
            P = x_conv[:,:x_conv.size(1)//2,:,:]
            Q = x_conv[:,x_conv.size(1)//2:,:,:]
            return (P + x_in) * torch.sigmoid(Q)
        else:
            return self.relu(x_conv + x_in)


# ================= CHEB GRAPH =================
class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, bias):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, gso):
        x = x.permute(0,2,3,1)
        B,T,N,C = x.shape
        x_list = [x]
        if self.Ks >= 2:
            x1 = torch.einsum('bij,btjc->btic', gso, x)
            x_list.append(x1)


        for k in range(2, self.Ks):
            xk = torch.einsum('bij,btjc->btic', 2*gso, x_list[-1]) - x_list[-2]
            x_list.append(xk)

        x = torch.stack(x_list, dim=2)
        out = torch.einsum('btkni,kio->btno', x, self.weight)

        if self.bias is not None:
            out += self.bias

        return out
      
# ================= GRAPH LAYER =================
class GraphConvLayer(nn.Module):
    def __init__(self, c_in, c_out, Ks, bias):
        super().__init__()
        self.align = Align(c_in, c_out)
        self.conv = ChebGraphConv(c_out, c_out, Ks, bias)


    def forward(self, x, gso):
        x_in = self.align(x)
        x_gc = self.conv(x_in, gso)
        x_gc = x_gc.permute(0,3,1,2)
        return x_gc + x_in
      
# ================= ST BLOCK =================
class STConvBlock(nn.Module):
    def __init__(self, Kt, Ks, n_vertex, last_block_channel,
                 channels, act_func, bias, droprate):
        super().__init__()

        self.tmp1 = TemporalConvLayer(Kt,last_block_channel,
                                      channels[0],n_vertex,act_func)
        self.graph = GraphConvLayer(
            channels[0],channels[1],Ks,bias
        )

        self.tmp2 = TemporalConvLayer(Kt,channels[1],
                                      channels[2],n_vertex,act_func)

        self.norm = nn.LayerNorm([n_vertex,channels[2]])
        self.dropout = nn.Dropout(droprate)

    def forward(self,x,gso):
        x = self.tmp1(x)
        x = self.graph(x,gso)
        x = self.tmp2(x)
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        return self.dropout(x)


# ================= OUTPUT =================
class OutputBlock(nn.Module):
    def __init__(self,Ko,last_block_channel,channels,end_channel,n_vertex,act_func,bias,droprate):
        super().__init__()
        self.tmp1=TemporalConvLayer(Ko,last_block_channel,channels[0],n_vertex,act_func)
        self.fc1=nn.Linear(channels[0],channels[1])
        self.fc2=nn.Linear(channels[1],end_channel*12)
        self.norm=nn.LayerNorm([n_vertex,channels[0]])
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(droprate)

    def forward(self,x):
        x=self.tmp1(x)
        x=self.norm(x.permute(0,2,3,1))
        x=self.relu(self.fc1(x))
        x=self.dropout(x)

        B,T,N,C=x.shape
        x=self.fc2(x).view(B,T,N,12)
        x=x[:,-1,:,:]   
        x=x.permute(0,2,1)


        return x
