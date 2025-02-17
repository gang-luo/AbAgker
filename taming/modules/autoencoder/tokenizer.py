import re
import torch

class Tokenizer_Pocket:
    def __init__(self):
        # 初始化特殊功能 token
        self.special_tokens = {
            "<cls>": 0,
            "<pad>": 1,
            "<eos>": 2,
            "<unk>": 3,
        }

        # 初始化蛋白质残基字符 token
        amino_acids = "LAGVSERTIDPKQNFYMHWCXBUZO"
        self.amino_acid_tokens = {amino_acids[i]: 4 + i for i in range(len(amino_acids))}

        # 初始化数字 token
        self.number_tokens = {str(i): i + 29 for i in range(1, 897)}

        # 合并所有 token
        self.token_to_id = {**self.special_tokens, **self.amino_acid_tokens, **self.number_tokens}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def tokenize(self, sequence,only_index):
        if only_index:
            tokens = sequence
        else:
            tokens = re.findall(r'[A-Za-z]|[0-9]+', sequence)
        
        return tokens

    def encode(self, sequences, max_len=None, padding=False, add_special_tokens=True, return_tensors=None,only_index=False):
        batch_token_ids = []
        for sequence in sequences:
            tokens = self.tokenize(sequence,only_index)
            if add_special_tokens:
                tokens = ['<cls>'] + tokens + ['<eos>']
            token_ids = [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens]

            if max_len:
                if len(token_ids) > max_len:
                    token_ids = token_ids[:max_len]
                elif padding and len(token_ids) < max_len:
                    token_ids = token_ids + [self.token_to_id['<pad>']] * (max_len - len(token_ids))

            batch_token_ids.append(token_ids)

        # 如果需要将 batch 数据转为相同长度，可以在此进行 padding
        if padding and max_len:
            for i, token_ids in enumerate(batch_token_ids):
                if len(token_ids) < max_len:
                    batch_token_ids[i] = token_ids + [self.token_to_id['<pad>']] * (max_len - len(token_ids))

        if return_tensors == "pt":
            return torch.tensor(batch_token_ids)
        else:
            return batch_token_ids

    def decode(self, batch_token_ids):
        batch_sequences = []
        for token_ids in batch_token_ids:
            tokens = [self.id_to_token.get(id, '<unk>') for id in token_ids]
            sequence = ''.join(tokens).replace('<pad>', '').replace('<cls>', '').replace('<eos>', '').replace('<unk>', '[UNK]')
            batch_sequences.append(sequence)
        return batch_sequences
