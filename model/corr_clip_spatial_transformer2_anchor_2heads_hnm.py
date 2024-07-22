import torch.nn as nn
import torch
import torch.nn.functional as F
from model.transformer import Block
from utils.model_utils import PositionalEncoding1D, positionalencoding1d, positionalencoding3d, positionalencoding2d
from utils.model_utils import BasicBlock_Conv2D, BasicBlock_MLP
from utils.anchor_utils import generate_anchor_boxes_on_regions
from dataset.dataset_utils import bbox_xyhwToxyxy
from einops import rearrange
import math
import torchvision
from dataset import dataset_utils
from model.mae import vit_base_patch16
import clip
import os
import random

base_sizes=torch.tensor([[16, 16], [32, 32], [64, 64], [128, 128]], dtype=torch.float32)    # 4 types of size
aspect_ratios=torch.tensor([0.5, 1, 2], dtype=torch.float32)                                # 3 types of aspect ratio
n_base_sizes = base_sizes.shape[0]
n_aspect_ratios = aspect_ratios.shape[0]


def build_backbone(config, with_text=False):
    name, bone_type = config.model.backbone_name, config.model.backbone_type
    if os.path.isdir(config.dataset.hub_dir):
        torch.hub.set_dir(config.dataset.hub_dir)
    if name == 'dino':
        assert bone_type in ['vitb8', 'vitb16', 'vits8', 'vits16']
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_{}'.format(bone_type))
        down_rate = int(bone_type.replace('vitb', '').replace('vits', ''))
        backbone_dim = 768
        if bone_type == 'vitb16' and config.model.bakcbone_use_mae_weight:
            mae_weight = torch.load('/vision/hwjiang/episodic-memory/VQ2D/checkpoint/mae_pretrain_vit_base.pth')['model']
            backbone.load_state_dict(mae_weight)
    elif name == 'dinov2':
        assert bone_type in ['vits14', 'vitb14', 'vitl14', 'vitg14']
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_{}'.format(bone_type))
        down_rate = 14
        if bone_type == 'vitb14':
            backbone_dim = 768
        elif bone_type == 'vits14':
            backbone_dim = 384
    elif name == 'mae':
        backbone = vit_base_patch16()
        cpt = torch.load('/vision/hwjiang/download/model_weight/mae_pretrain_vit_base.pth')['model']
        backbone.load_state_dict(cpt, strict=False)
        down_rate = 16
        backbone_dim = 768
    elif name in ['CLIP', 'CLIP_text']:
        if os.path.isfile(config.model.clip_dir):
            backbone, _ = clip.load(config.model.clip_dir, device='cuda') 
        else:
            backbone, _ = clip.load(bone_type, device='cuda') 
        backbone_dim = 768
        down_rate = backbone.visual.conv1.kernel_size[0]
        idal_patches = (config.dataset.clip_size_fine // down_rate)**2+1
        if idal_patches != backbone.visual.positional_embedding.shape[0]:
            backbone.visual.positional_embedding = interpolate_pos_encoding(backbone.visual.positional_embedding, patches=idal_patches, dim= backbone_dim, height=config.dataset.clip_size_fine, width=config.dataset.clip_size_coarse, patch_size=backbone.visual.conv1.kernel_size)

    if with_text:
        text_name = config.model.text_backbone_name
        if text_name == 'CLIP':
            text_backbone_dim = 512
            if name == 'CLIP':
                text_backbone = backbone
            else:
                if os.path.isfile(config.model.clip_dir):
                    text_backbone, _ = clip.load(config.model.clip_dir, device='cuda') 
                else:
                    text_backbone, _ = clip.load('ViT-B/16', device='cuda') 
                # text_backbone.visual = nn.Identity()
        return backbone, text_backbone, down_rate, backbone_dim, text_backbone_dim
            
    return backbone, None, down_rate, backbone_dim, None

def interpolate_pos_encoding(positional_embedding, patches: int, dim: int, height: int, width: int, patch_size) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
    resolution images.
    Source:
    https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
    """

    num_patches = patches - 1
    pos_embedding = positional_embedding.unsqueeze(0)
    num_positions = pos_embedding.shape[1] - 1
    class_pos_embed = pos_embedding[:, 0]
    patch_pos_embed = pos_embedding[:, 1:]
    h0 = height // patch_size[0]
    w0 = width // patch_size[1]
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    output = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    return torch.nn.Parameter(output.squeeze(0)).to(positional_embedding.device)

def freeze_backbone(model, model2=None):
    for param in model.parameters():
        param.requires_grad = False
    if model2 is not None:
        for param in model2.parameters():
            param.requires_grad = False
        

class ClipMatcher(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.query_type = config.model.query_type # 448, 224
        assert self.query_type in ['image', 'text', 'both']
        self.with_text = True if self.query_type in ['text','both'] else False
        self.backbone, self.text_backbone, self.down_rate, self.backbone_dim, self.text_backbone_dim = build_backbone(config, self.with_text)
        self.backbone_name, self.text_backbone_name = config.model.backbone_name, config.model.text_backbone_name
        if config.model.fix_backbone:
            freeze_backbone(self.backbone, self.text_backbone)
        
        self.query_size = config.dataset.query_size # 448, 224
        self.clip_size_fine = config.dataset.clip_size_fine # 448, 224
        self.clip_size_coarse = config.dataset.clip_size_coarse # 448, 224

        self.query_feat_size = self.query_size // self.down_rate # 32
        self.clip_feat_size_fine = self.clip_size_fine // self.down_rate # 32
        self.clip_feat_size_coarse = self.clip_size_coarse // self.down_rate # 32

        self.CQ_after_reduce = config.model.CQ_after_reduce
        self.type_transformer = config.model.type_transformer
        assert self.type_transformer in ['local', 'global']
        self.window_transformer = config.model.window_transformer
        self.resolution_transformer = config.model.resolution_transformer
        self.resolution_anchor_feat = config.model.resolution_anchor_feat

        self.anchors_xyhw = generate_anchor_boxes_on_regions(image_size=[self.clip_size_coarse, self.clip_size_coarse],
                                                        num_regions=[self.resolution_anchor_feat, self.resolution_anchor_feat])
        self.anchors_xyhw = self.anchors_xyhw / self.clip_size_coarse   # [R^2*N*M,4], value range [0,1], represented by [c_x,c_y,h,w] in torch axis
        self.anchors_xyxy = bbox_xyhwToxyxy(self.anchors_xyhw)
        
        # query down heads ???????????????
        self.query_down_heads = []
        for _ in range(int(math.log2(self.query_feat_size))):
            self.query_down_heads.append(
                nn.Sequential(
                    nn.Conv2d(self.backbone_dim, self.backbone_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(self.backbone_dim),
                    nn.LeakyReLU(inplace=True),
                )
            )
        self.query_down_heads = nn.ModuleList(self.query_down_heads)

        # if self.query_type in ['text','both']:
        # feature reduce layer
        self.reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        
        if self.query_type in ['text','both']:
            if not self.CQ_after_reduce:
                self.text_reduce = nn.Sequential(
                    nn.Conv1d(self.text_backbone_dim, 256, 3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv1d(256, 256, 3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(inplace=True),
                )
            else:
                scale = self.backbone_dim ** -0.5
                self.text_proj = nn.Parameter(scale * torch.randn(self.text_backbone_dim, self.backbone_dim))
        
        d_dim = 256 if not self.CQ_after_reduce else 768
        
        # clip-query correspondence
        self.CQ_corr_transformer = []
        for _ in range(1):
            self.CQ_corr_transformer.append(
                torch.nn.TransformerDecoderLayer(
                    d_model=d_dim,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True
                )
            )
        self.CQ_corr_transformer = nn.ModuleList(self.CQ_corr_transformer)

        # feature downsample layers
        self.num_head_layers, self.down_heads = int(math.log2(self.clip_feat_size_coarse)), []
        for i in range(self.num_head_layers-1):
            self.in_channel = 256 if i != 0 else self.backbone_dim
            self.down_heads.append(
                nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
            ))
        self.down_heads = nn.ModuleList(self.down_heads)

        # spatial-temporal PE
        self.pe_3d = positionalencoding3d(d_model=256, 
                                          height=self.resolution_transformer, 
                                          width=self.resolution_transformer, 
                                          depth=config.dataset.clip_num_frames,
                                          type=config.model.pe_transformer).unsqueeze(0)
        self.pe_3d = nn.parameter.Parameter(self.pe_3d)

        # spatial-temporal transformer layer
        self.feat_corr_transformer = []
        self.num_transformer = config.model.num_transformer
        for _ in range(self.num_transformer):
            self.feat_corr_transformer.append(
                    torch.nn.TransformerEncoderLayer(
                        d_model=256, 
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.0,
                        activation='gelu',
                        batch_first=True
                ))
        self.feat_corr_transformer = nn.ModuleList(self.feat_corr_transformer)
        self.temporal_mask = None

        # output head
        self.head = Head(in_dim=256, in_res=self.resolution_transformer, out_res=self.resolution_anchor_feat)

    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def extract_feature(self, x, return_h_w=True, enable_proj=False):
        if self.backbone_name == 'dino':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token   # [b, h*w, c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size), int(w_origin / self.backbone.patch_embed.patch_size)
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name == 'dinov2': # ours
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1]) # 448 / 14 = 32
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2) # [B, D, H, W]
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name == 'mae':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.forward_features(x) # [b,1+h*w,c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
            dim = out.shape[-1]
            out = out[:,1:].reshape(b, h, w, dim).permute(0,3,1,2)  # [b,c,h,w]
            out = F.interpolate(out, size=(16,16), mode='bilinear')
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name in ['CLIP', 'CLIP_text']:
            x = x.half()
            b, _, h_origin, w_origin = x.shape # h_origin, w_origin -> 224
            visual = self.backbone.visual
            out = self.forward_clip(x, visual, enable_proj) # [3, 49, 768]
            patch_size = int(self.config.model.backbone_type.replace('ViT-B/', '')) # 32
            h, w = int(h_origin / patch_size), int(w_origin / patch_size) # 7, 7
            dim = out.shape[-1] # 768
            out = out.reshape(b, h, w, dim).permute(0,3,1,2).float()
            if return_h_w:
                return out, h, w
            return out
        
    def encode_text(self, text):
        device = next(self.text_backbone.parameters()).device
        x = self.text_backbone.token_embedding(text.to(device)).type(self.text_backbone.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.text_backbone.positional_embedding.type(self.text_backbone.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_backbone.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.text_backbone.ln_final(x).type(self.text_backbone.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_backbone.text_projection
        x = x[torch.arange(x.shape[0])] @ self.text_backbone.text_projection

        return x
    
    def extract_text_feature(self, x):
        if self.text_backbone_name == 'CLIP':
            text = clip.tokenize(x)
            text_features = self.encode_text(text)
            text_features = text_features.permute(0,2,1).float()
            # text_features = (text_features.permute(0,2,1) @ self.text_proj.to(text_features.dtype)).permute(0,2,1)
        return text_features
        
    def forward_clip(self, x, visual, enable_proj=False):
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1) # shape = [*, grid ** 2 + 1, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_post(x[:, 0, :])
        x = visual.ln_post(x[:, 1:, :])
        
        if enable_proj:
            x = x @ visual.proj
        return x
    
    def replicate_for_hnm(self, query_feat, clip_feat, equal_negative, query_text_feat=None):
        '''
        query_feat in shape [b,c,h,w]
        clip_feat in shape [b*t,c,h,w]
        '''
        b = query_feat.shape[0]
        bt = clip_feat.shape[0]
        t = bt // b
        
        clip_feat = rearrange(clip_feat, '(b t) c h w -> b t c h w', b=b, t=t)

        new_clip_feat, new_query_feat, new_query_text_feat = [], [], []
        if not equal_negative:
            for i in range(b):
                for j in range(b):
                    new_clip_feat.append(clip_feat[i])
                    new_query_feat.append(query_feat[j])
                    if query_text_feat:
                        new_query_text_feat.append(query_text_feat[j])
                        
        else:
            negative_indices = []
            for i in range(b):
                new_clip_feat.append(clip_feat[i])
                new_query_feat.append(query_feat[i])
                if query_text_feat:
                    new_query_text_feat.append(query_text_feat[j])
                for j in range(b):
                    if i != j:
                        negative_indices.append((i, j))
            negative_indices = random.sample(negative_indices, b)
            for i, j in negative_indices:
                new_clip_feat.append(clip_feat[i])
                new_query_feat.append(query_feat[j])
                if query_text_feat:
                    new_query_text_feat.append(query_text_feat[j])

        new_clip_feat = torch.stack(new_clip_feat)      # [b^2,t,c,h,w]
        new_query_feat = torch.stack(new_query_feat)    # [b^2,c,h,w]
        if query_text_feat:
            new_query_text_feat = torch.stack(new_query_text_feat)    # [b^2,c,h,w]

        new_clip_feat = rearrange(new_clip_feat, 'b t c h w -> (b t) c h w')
        return new_clip_feat, new_query_feat, new_query_text_feat


    def forward(self, clip, query, query_text=None, query_frame_bbox=None, training=False, fix_backbone=True):
        '''
        clip: in shape [b,t,c,h,w]
        query: in shape [b,c,h2,w2]
        '''
        b, t = clip.shape[:2] # t = 30 -> clip_num_frames
        clip = rearrange(clip, 'b t c h w -> (b t) c h w')

        # get backbone features
        if fix_backbone:
            with torch.no_grad():
                query_text_feat = None
                clip_feat, h, w = self.extract_feature(clip) # (b t) c h w -> [b*30(clip_num_frames), 768, 32, 32]
                if self.query_type in ['text','both']:
                    query_text_feat = self.extract_text_feature(query_text)
                    # clip_text_feat = (clip_feat.permute(0,2,3,1) @ self.backbone.visual.proj).permute(0,3,1,2)
                # if self.query_type in ['image','both']:
                if self.query_type in ['image','both','text']:
                    query_feat, _, _ = self.extract_feature(query) # [b c h w] -> [b, 768, 32, 32]
        else:
            query_text_feat = None
            clip_feat, h, w = self.extract_feature(clip)
            if self.query_type in ['text','both']:
                query_text_feat = self.extract_text_feature(query_text)
                # clip_text_feat = (clip_feat.permute(0,2,3,1) @ self.backbone.visual.proj).permute(0,3,1,2)
            # if self.query_type in ['image','both']:
            if self.query_type in ['image','both','text']:
                query_feat, _, _ = self.extract_feature(query)
    
        # h, w = clip_feat.shape[-2:] # dinov2 -> 32, 32

        if torch.is_tensor(query_frame_bbox) and self.config.train.use_query_roi:
            idx_tensor = torch.arange(b, device=clip.device).float().view(-1, 1)
            query_frame_bbox = dataset_utils.recover_bbox(query_frame_bbox, h, w)
            roi_bbox = torch.cat([idx_tensor, query_frame_bbox], dim=1)
            query_feat = torchvision.ops.roi_align(query_feat, roi_bbox, (h,w))

        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        
        if not self.CQ_after_reduce:
            # reduce channel size
            all_feat = self.reduce(all_feat)
        if self.query_type in ['text','both']:
            if not self.CQ_after_reduce:
                query_text_feat = self.text_reduce(query_text_feat)
            else:
                query_text_feat = (query_text_feat.permute(0,2,1) @ self.text_proj.to(query_text_feat.dtype)).permute(0,2,1)
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)
        query_text_feat = None if query_text_feat is None else query_text_feat

        if (self.config.train.use_hnm or self.config.train.use_fix_hnm) and training:
            clip_feat, query_feat, query_text_feat = self.replicate_for_hnm(query_feat, clip_feat, self.config.train.use_fix_hnm, query_text_feat)   # b -> b^2
            b = b**2 if not self.config.train.use_fix_hnm else b*2
        
        # find spatial correspondence between query-frame
        if self.query_type == 'text':
            query_feat = rearrange(query_text_feat.unsqueeze(1).repeat(1,t,1,1), 'b t c n -> (b t) n c')      # [b*t,n,c]
        elif self.query_type == 'both':
            query_feat = rearrange(query_feat.unsqueeze(1).repeat(1,t,1,1,1), 'b t c h w -> (b t) (h w) c')   # [b*t,n,c]
            query_text_feat = rearrange(query_text_feat.unsqueeze(1).repeat(1,t,1,1), 'b t c n -> (b t) n c') # [b*t,n,c]
            query_feat = torch.concat([query_feat,query_text_feat], dim=1)
        else:
            query_feat = rearrange(query_feat.unsqueeze(1).repeat(1,t,1,1,1), 'b t c h w -> (b t) (h w) c')   # [b*t,n,c]
            
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')                                              # [b*t,n,c]
        
        # spatial correspondence
        for layer in self.CQ_corr_transformer:
            clip_feat = layer(clip_feat, query_feat)                                                          # [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b (h w) c -> b c h w', h=h, w=w)                                    # [b*t,c,h,w] # [90, 256, 32, 32]
        
        if self.CQ_after_reduce:
            # reduce channel size
            clip_feat = self.reduce(clip_feat)

        # down-size features and find spatial-temporal correspondence
        for head in self.down_heads:
            clip_feat = head(clip_feat) # [b, 30 * 8 * 8, 256]
            if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
                clip_feat = rearrange(clip_feat, '(b t) c h w -> b (t h w) c', b=b) + self.pe_3d
                mask = self.get_mask(clip_feat, t)
                for layer in self.feat_corr_transformer:
                    clip_feat = layer(clip_feat, src_mask=mask)
                clip_feat = rearrange(clip_feat, 'b (t h w) c -> (b t) c h w', b=b, t=t, h=self.resolution_transformer, w=self.resolution_transformer)
                break
        
        # refine anchors
        anchors_xyhw = self.anchors_xyhw.to(clip_feat.device)                   # [N,4]
        anchors_xyxy = self.anchors_xyxy.to(clip_feat.device)                   # [N,4]
        anchors_xyhw = anchors_xyhw.reshape(1,1,-1,4)                           # [1,1,N,4]
        anchors_xyxy = anchors_xyxy.reshape(1,1,-1,4)                           # [1,1,N,4]
        
        # head
        bbox_refine, prob = self.head(clip_feat)                                # [b*t,N=h*w*n*m,c]
        bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  # [b,t,N,4], in xyhw frormulation
        prob = rearrange(prob, '(b t) N c -> b t N c', b=b, t=t)                # [b,t,N,1]
                                                     # [b,t,N,4]

        center, hw = bbox_refine.split([2,2], dim=-1)                           # represented by [c_x, c_y, h, w]
        hw = 0.5 * hw                                                           # anchor's hw is defined as real hw
        bbox = torch.cat([center - hw, center + hw], dim=-1)                    # [b,t,N,4]

        result = {
            'center': center,           # [b,t,N,2]
            'hw': hw,                   # [b,t,N,2]
            'bbox': bbox,               # [b,t,N,4]
            'prob': prob.squeeze(-1),   # [b,t,N]
            'anchor': anchors_xyxy      # [1,1,N,4] ############
        }
        return result
    

    def get_mask(self, src, t):
        if not torch.is_tensor(self.temporal_mask):
            hw = src.shape[1] // t
            thw = src.shape[1]
            mask = torch.ones(thw, thw).float() * float('-inf')

            window_size = self.window_transformer // 2

            for i in range(t):
                min_idx = max(0, (i-window_size)*hw)
                max_idx = min(thw, (i+window_size+1)*hw)
                mask[i*hw: (i+1)*hw, min_idx: max_idx] = 0.0
            mask = mask.to(src.device)
            self.temporal_mask = mask
        return self.temporal_mask
    


class Head(nn.Module):
    def __init__(self, in_dim=256, in_res=8, out_res=16, n=n_base_sizes, m=n_aspect_ratios):
        super(Head, self).__init__()

        self.in_dim = in_dim
        self.n = n
        self.m = m
        self.num_up_layers = int(math.log2(out_res // in_res))
        self.num_layers = 3
        
        if self.num_up_layers > 0:
            self.up_convs = []
            for _ in range(self.num_up_layers):
                self.up_convs.append(torch.nn.ConvTranspose2d(in_dim, in_dim, kernel_size=4, stride=2, padding=1))
            self.up_convs = nn.Sequential(*self.up_convs)

        self.in_conv = BasicBlock_Conv2D(in_dim=in_dim, out_dim=2*in_dim)

        self.regression_conv = []
        for i in range(self.num_layers):
            self.regression_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.regression_conv = nn.Sequential(*self.regression_conv)

        self.classification_conv = []
        for i in range(self.num_layers):
            self.classification_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.classification_conv = nn.Sequential(*self.classification_conv)

        self.droupout_feat = torch.nn.Dropout(p=0.2)
        self.droupout_cls = torch.nn.Dropout(p=0.2)

        self.regression_head = nn.Conv2d(in_dim, n * m * 4, kernel_size=3, padding=1)
        self.classification_head = nn.Conv2d(in_dim, n * m * 1, kernel_size=3, padding=1)

        self.regression_head.apply(self.init_weights_conv)
        self.classification_head.apply(self.init_weights_conv)

    def init_weights_conv(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def forward(self, x):
        '''
        x in shape [B,c,h=8,w=8]
        '''
        if self.num_up_layers > 0:
            x = self.up_convs(x)     # [B,c,h=16,w=16]

        B, c, h, w = x.shape

        feat_reg, feat_cls = self.in_conv(x).split([c, c], dim=1)   # both [B,c,h,w]
        # dpout pos 1, seems better
        feat_reg = self.droupout_feat(feat_reg)
        feat_cls = self.droupout_cls(feat_cls)

        feat_reg = self.regression_conv(feat_reg)        # [B,n*m*4,h,w]
        feat_cls = self.classification_conv(feat_cls)    # [B,n*m*1,h,w]

        # dpout pos 2

        out_reg = self.regression_head(feat_reg) # box
        out_cls = self.classification_head(feat_cls) # probability

        out_reg = rearrange(out_reg, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=4)
        out_cls = rearrange(out_cls, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=1)

        return out_reg, out_cls
