import torch
from torch import nn
from .transformer import DecoderLayer

class M_decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, num_layers: int, m_class:int ):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, ff_dim) for _ in range(num_layers)])
        self.start_token_id = torch.tensor([0], dtype=torch.long) # 初始化为张量 
        self.closed_token_id = torch.tensor([1], dtype=torch.long) # 初始化为张量
        self.fc_m_class = nn.Linear(d_model, m_class)
        
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def forward(self, encoder_inputs, target_inputs,m_embeding_layer,padding_mask=None, max_len=128):
        batch_size = encoder_inputs.size(0)
                                         
        if target_inputs is not None:
            tgt_mask = self.generate_square_subsequent_mask(target_inputs.size(1)).to(encoder_inputs.device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
            # combined_mask = tgt_mask | padding_mask
            memory_mask = None

            for layer in self.layers:
                target_inputs = layer(target_inputs, encoder_inputs, tgt_mask, memory_mask) # 不使用附带padding mask的combined_mask
            outputs = self.fc_m_class(target_inputs)
            # add
            outputs = F.softmax(outputs, dim = -1)
            return outputs
        else:
            self.start_token_id = self.start_token_id.to(encoder_inputs.device)
            self.closed_token_id = self.closed_token_id.to(encoder_inputs.device)
            generated = m_embeding_layer(self.start_token_id).unsqueeze(0).expand(batch_size, -1, -1)

            for _ in range(max_len - 1):
                tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(encoder_inputs.device)
                tgt_mask = tgt_mask.unsqueeze(0).expand(generated.size(0), -1, -1)

                target_inputs = generated
                for layer in self.layers:
                    target_inputs = layer(target_inputs, encoder_inputs, tgt_mask, None)
                target_inputs = self.fc_m_class(target_inputs[:, -1:, :])
                # add
                target_inputs = F.softmax(target_inputs, dim = -1)
                # 如果需要进行多样化生成，则需要计算softmax并采用温度系数进行采用，生成多样化分子
                next_token_id = torch.argmax(target_inputs, dim=-1)  
                generated = torch.cat((generated, m_embeding_layer(next_token_id)), dim=1)

                # 动态中止条件
                if (next_token_id == self.closed_token_id).all():
                    break
                
            return self.fc_m_class(generated)