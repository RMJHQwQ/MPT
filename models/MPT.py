import torch

import torch.nn as nn
from models.qformer import QFormer
from models.utils import CrossAttention, SKNetandLSTM_Model, AxModel, SKNet_LSTM_Attention_Model
# torch.cuda.manual_seed(42)
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sigmoid()  # Swish uses sigmoid

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * self.gate(x2)

class MPT(nn.Module):
    def __init__(self,batch_size,basemodel,visual_model):
        super().__init__()
        self.batch_size = batch_size
        self.base_model = basemodel
        self.visual_model = visual_model
        self.qformer = QFormer(dim=768, heads=2, depth=6, dropout=0.1, text_block_depth=1, img_text_block_depth=1)
        self.qformer.to("cuda")
        self.img_to_txt_att = CrossAttention(dim=768, heads=1, dropout=0.1)
        self.llm_tensor_att = CrossAttention(dim=768, heads=1, dropout=0.1)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Tanh()
        self.adjust_visual = nn.Linear(768, 768*320)
        self.SwiGlu = SwiGLU()
        self.downsample = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 768*4),
            self.SwiGlu,
            nn.Linear(768*2, 768),
            nn.Dropout(0.2),
        )
        for param in basemodel.parameters():
            param.requires_grad = (True)
        self.axmodel = AxModel(basemodel)
        self.pemnet_txt = SKNetandLSTM_Model(self.base_model, 3, 768)
        self.classf = nn.Linear(768,3)
        self.projj = nn.Linear(1536,768)
        self.GELU = nn.GELU()
        self.llm_weight = nn.Parameter(torch.tensor(0.5))
        self.laff_weight = nn.Parameter(torch.tensor(0.5))


    def LAFF(self, txt_f, img_f):
        Tanh = nn.Tanh()
        Tanh = Tanh.to("cuda")
        activated_txt = Tanh(txt_f)
        activated_img = Tanh(img_f)
        activated_txt = activated_txt.to("cuda")
        activated_img = activated_img.to("cuda")
        linear = nn.Linear(768, 1)
        activated_txt = self.dropout(activated_txt)
        activated_img = self.dropout(activated_img)
        linear = linear.to("cuda")
        activated_txt = activated_txt.unsqueeze(1)
        activated_img = activated_img.unsqueeze(1)
        fusion = torch.concat((activated_txt, activated_img), dim=1)
        out = linear(fusion)
        softmax = torch.nn.Softmax(dim=1)
        out = softmax(out)
        final = torch.mul(fusion, out)
        final = torch.mean(final, dim=1)
        return final



    # def forward(self, input, img, pos, neu, neg):
    #     x = self.base_model(**input)
    #     tokens = x.last_hidden_state
    
    #     visual_model_out = self.visual_model(img)
    #     visual_model_out = self.adjust_visual(visual_model_out)
    #     visual_model_out = visual_model_out.reshape(img.shape[0],-1,768)
    #     txt_q = self.img_to_txt_att(visual_model_out, tokens)
    #     tokens.to("cuda")
    #     img.to("cuda")
    #     llm = self.axmodel(pos, neu, neg)
    #     out, txt_out = self.qformer(tokens, img)
    #     # out = torch.mean(out,dim=1)
    #     # txt_q = torch.mean(txt_q,dim=1)
    #     # txt_out = x.last_hidden_state[:, 0, :]
    #     combine = torch.concat((out,txt_q),dim=-1)
    #     # combine = self.act(combine)
        
    #     # print(llm.shape)
    #     # print(combine.shape)
    #     combine = self.projj(combine)
    #     combine = self.GELU(combine)
    #     dop = self.llm_tensor_att(llm, combine )
    #     dop = torch.mean(dop,dim=1)
    #     dop = self.dropout(combine)
    #     down_out = self.downsample(dop)
        
        
    #     down_out = torch.mean(down_out,dim=1)
    #     # print(down_out.shape)
    #     # out = self.softmax(down_out)
    #     return self.classf(down_out)

    def forward(self, input, img, pos, neu, neg):
        txt_f = self.pemnet_txt(input)
        img_f = self.visual_model(img)
        f = self.LAFF(txt_f, img_f)
        llm = self.axmodel(pos, neu, neg)
        # llm = torch.mean(llm,dim=1)
        
        # pre = pre.unsqueeze(dim=1)
        # pre = self.llm_tensor_att(pre, llm)

        
        llm_cls = torch.mean(llm,dim=1)  # 取每条样本第一个 token 的表示
        fused = self.llm_weight * llm_cls + self.laff_weight * f
        # fused = f

        
        pre = self.downsample(fused) + fused
        # pre = pre.mean(dim=1)
        res = self.classf(pre)
        # pre = pre.squeeze(dim=1)
        pre.to("cuda")
        return res, fused


