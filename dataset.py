import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,random_split

class BillingualDataset(Dataset):
    def __init__(self,ds,tokenizer_src,tokenizer_tgt,src_lang,tgt_lang,seq_len):
        super().__init__()
        self.ds=ds
        self.tokenizer_src=tokenizer_src
        self.tokenizer_tgt=tokenizer_tgt
        self.src_lang=src_lang
        self.tgt_lang=tgt_lang
        self.seq_len=seq_len
        
        self.sos_token=torch.tensor([tokenizer_tgt.piece_to_id('[SOS]')],dtype=torch.int64)
        self.eos_token=torch.tensor([tokenizer_tgt.piece_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token=torch.tensor([tokenizer_tgt.piece_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair=self.ds[index]
        src_text=src_target_pair['translation'][self.src_lang]
        tgt_text=src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text, out_type=int)
        dec_input_tokens=self.tokenizer_tgt.encode(tgt_text,out_type=int)

        enc_num_padding_tokens=self.seq_len-len(enc_input_tokens)-2 # -2 for removing sos and eos 
        dec_num_padding_tokens=self.seq_len-len(dec_input_tokens)-1 # -1 for removing eos 
        if enc_num_padding_tokens<0 or dec_num_padding_tokens<0:
            raise ValueError('sentance is too long')
        
        encoder_input=torch.cat(
            [self.sos_token,
            torch.tensor(enc_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]* enc_num_padding_tokens,dtype=torch.int64)
            ],dim=0
        )

        decoder_input=torch.cat(
            [ self.sos_token,
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            torch.tensor([self.pad_token]* dec_num_padding_tokens,dtype=torch.int64)
            ],dim=0
        )
        
        # what we expect as the output of the decoder 
        label=torch.cat(
            [
            torch.tensor(dec_input_tokens,dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token]* dec_num_padding_tokens,dtype=torch.int64)
            ],dim=0
        )
        
        assert encoder_input.size(0)==self.seq_len
        assert decoder_input.size(0)==self.seq_len
        assert label.size(0)==self.seq_len

        return {
            'encoder_input':encoder_input,
            'decoder_input':decoder_input,
            'encoder_mask':(encoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int(), #(1,1,seq_len)
            'decoder_mask':(decoder_input!=self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'label':label,
            'src_text':src_text,
            'tgt_text':tgt_text
        }

def causal_mask(size):
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask==0

    
      