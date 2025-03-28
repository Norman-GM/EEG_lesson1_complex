#  https://github.com/snailpt/CTNet/blob/main/Conformer_fine_tune_2a_77.66_2b_85.87.ipynb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from torch import Tensor
from codes.models.ModelResgistry import MODEL_REGISTOR, BaseModel

class Conformer(nn.Module):
    def __init__(self, number_channel=22, nb_classes=4, dropout_rate=0.5):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (number_channel, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.trans = TransformerEncoder(10, 6, 40)
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        #         b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        x = self.trans(x)
        x = self.flatten(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, num_heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads) for _ in range(depth)])

class Loss_fn(nn.Module):
    def __init__(self):
        super(Loss_fn, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        loss = self.loss(outputs, labels)
        return loss


@MODEL_REGISTOR.register(loss_fn=Loss_fn()) # Register the model with loss function
class EEG_conformer_CTN(BaseModel):
    def __init__(self, heads=4,
                 emb_size=40,
                 depth=6,
                 database_type='A',
                 eeg1_f1=20,
                 eeg1_kernel_size=64,
                 eeg1_D=2,
                 eeg1_pooling_size1=8,
                 eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3,
                 eeg1_number_channel=22,
                 flatten_eeg1=600,
                 **kwargs):
        super().__init__()
        self.number_class, self.number_channel = 4,22
        self.flatten_eeg1 = flatten_eeg1
        self.net = Conformer(number_channel=self.number_channel, nb_classes=self.number_class)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2440, self.number_class)
        )

    def forward(self, x):
        x = self.net(x)
        out = self.fc(x)
        return out