from torch import nn
from models.vision_transformer import VisionTransformer
from models.decoder import Decoder
#from models.attentive_pooler import AttentivePooler


class AVGPOOLMODEL(nn.Module):
    def __init__(self,
                 img_size = 182,
                 patch_size = 14,
                 num_frames = 224,
                 tubelet_size=16,
                 in_chans = 1,
                 embed_dim=384,
                 depth=12,
                 #num_queries=10,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 init_std=0.02,
                 out_layers=None,
                 uniform_power=False,
                 non_zero_patch_opt=True,
                 decoder_embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.image_size = img_size
        self.num_frames = num_frames
        #self.num_queries = num_queries
        #self.embed_dim = embed_dim
        #self.decoder_embed_dim = decoder_embed_dim
        self.grid_size = self.image_size // self.patch_size
        self.grid_depth = self.num_frames // self.tubelet_size
        self.encoder = VisionTransformer(img_size, patch_size, num_frames, tubelet_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate, norm_layer, init_std, out_layers, non_zero_patch_opt)
        #self.attn_pooler = AttentivePooler(num_queries = num_queries, embed_dim = embed_dim)
        self.decoder = Decoder(embed_dim = embed_dim, decoder_embed_dim = decoder_embed_dim, num_heads = num_heads)

    def forward_loss(self, pred, imgs, batch_mask, non_zero_mask):
        target = imgs.view(\
        (-1, self.grid_size, self.patch_size, self.grid_depth, self.tubelet_size,\
         self.grid_size, self.patch_size)).permute(0,1,3,5,2,4,6).reshape((-1, 13*14*13, 14*16*14))
        #target = imgs.view((-1, self.grid_size, self.patch_size, self.grid_depth, self.tubelet_size, self.grid_size, self.patch_size)).permute(0, 2, 4, 6, 1, 3, 5).view((-1, self.grid_size * self.grid_depth * self.tubelet_size, self.patch_size ** 2 * self.grid_depth))
        target = target[:, batch_mask, :]
        mask = non_zero_mask[:, batch_mask]
        loss = ((target - pred)**2).mean(-1)
        #loss = (target - pred)**2)
        return (loss * mask).sum() / mask.sum()
        #return (loss * mask).sum(1) / mask.sum() ## CHECK THIS OUT LATER FOR INDIVIDUAL RECON LOSS
    
    def forward(self, imgs, y):
        latent, batch_mask, non_zero_mask = self.encoder(imgs, y)
        #compute_attn = self.attn_pooler(latent)
        compute_pool = latent.mean(1, keepdim=True)
        #pred = self.decoder(compute_attn, batch_mask)
        pred = self.decoder(compute_pool, batch_mask)
        loss = self.forward_loss(pred, imgs, batch_mask, non_zero_mask)
        #return compute_attn, loss, pred, batch_mask
        return compute_pool, loss, pred, batch_mask
        
        
        
