import torch
from torch import nn
import math
class InputEmbedding(nn.Module):
    def __init__(self,vocab_size:int,d_model:int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    def __init__(self,d_model:int,sequence_length:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.sequence_length=sequence_length
        # self.Pos_embedding=nn.Embedding(vocab_size,d_model)
        self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(sequence_length,d_model) #(T,n_embd)
        pos=torch.arange(0,sequence_length,dtype=torch.float).unsqueeze(1) #(1,T,n_embd)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(pos*div_term)
        pe[:,1::2]=torch.cos(pos*div_term)
        pe=pe.unsqueeze(0) #(1,T,n_embd)

        self.register_buffer('pe',pe)

    def forward(self,x):

        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False) #token_embd.shape[1] is sequence length or T 
        return self.dropout(x)


    
class FeedForward(nn.Module):
    def __init__(self,d_model:int,dropout:float,ff=4):
        super().__init__()
        self.d_model=d_model
        self.ffw=nn.Sequential(
            nn.Linear(d_model,ff*d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff*d_model,d_model),
        )
    
    def forward(self,x):
        return self.ffw(x)


class MultiHeadAttention(nn.Module):
    def __init__(self,d_model:int,n_head:int,dropout:float):
        super().__init__()
        self.d_model=d_model
        self.h=n_head
        assert d_model%n_head==0,' d_model is  not divisible by the no. of heads'

        self.dk=d_model//self.h
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self.w_o=nn.Linear(d_model,d_model,bias=False) #project layer pf multihead attentiion'
        

    @staticmethod
    def Attention(query,key,value,mask,dropout:nn.Dropout):
        dk=query.shape[-1]

        attention_scores=(query @ key.transpose(-1,-2))/math.sqrt(dk) # (B,h,T,d_k)-->(B,h,T,T)

        if  mask is not None:
            # print("Attention scores shape:", attention_scores.shape)
            # print("Mask shape:", mask.shape)
            attention_scores.masked_fill_(mask==0,-1e9)

        attention_scores=attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores=dropout(attention_scores)

        return (attention_scores @ value),attention_scores


    
    def forward(self,q,k,v,mask):
        key=self.w_k(k)      # (B,T,n_emd)-->(B,T,n_emd)
        query=self.w_q(q)    # (B,T,n_emd)-->(B,T,n_emd)
        value=self.w_v(v)    # (B,T,n_emd)-->(B,T,n_emd)

        query=query.view(query.shape[0],query.shape[1],self.h,self.dk).transpose(1,2)# (B,T,n_emd)-->(B,T,h,d_k)-->(B,h,T,d_k)
        key=key.view(key.shape[0],key.shape[1],self.h,self.dk).transpose(1,2)        # (B,T,n_emd)-->(B,T,h,d_k)-->(B,h,T,d_k)
        value=value.view(value.shape[0],value.shape[1],self.h,self.dk).transpose(1,2)# (B,T,n_emd)-->(B,T,h,d_k)-->(B,h,T,d_k)


        x,self.attention_scores=MultiHeadAttention.Attention(query,key,value,mask,self.dropout)

        #(B,h,T,d_k)-->(B,T,h,d_k)-->(B,T,d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.dk)   #self.h*self.dk=d_model
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self,dropout:float,d_model=512):
        super().__init__()
       
        self.dropout = nn.Dropout(dropout)
        self.norm=nn.LayerNorm(d_model)
        

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self,self_multihead_attention:MultiHeadAttention,feed_foward:FeedForward,dropout:float):
        super().__init__()
       
        self.self_multihead_attention = self_multihead_attention
        self.feed_forward=feed_foward
        self.dropout=dropout
        # range=2  as encoder block only consits of 2 residual connections
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) 
        
        

    def forward(self,x,src_mask):
        x=self.residual_connections[0](x,lambda x:self.self_multihead_attention(x,x,x,src_mask))
        x=self.residual_connections[1](x,self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self,layers:nn.ModuleList,d_model=512):
        super().__init__()
       
        self.layers = layers
        self.norm=nn.LayerNorm(d_model)

    def forward(self,x,mask):
        for  layer in self.layers:
            x=layer(x,mask)

        return self.norm(x)



class DecoderBlock(nn.Module):
    def __init__(self,self_multihead_attention:MultiHeadAttention,cross_multihead_attention:MultiHeadAttention,feed_foward:FeedForward,dropout:float):
        super().__init__()
       
        self.self_multihead_attention = self_multihead_attention
        self.cross_multihead_attention = cross_multihead_attention
        self.feed_forward=feed_foward
        self.dropout=dropout
        # range=3  as decoder block consits of 3 residual connections
        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)]) 
        
        

    def forward(self,x,encoder_ouput,src_mask,tgt_mask):
        x=self.residual_connections[0](x,lambda x:self.self_multihead_attention(x,x,x,tgt_mask))
        x=self.residual_connections[1](x,lambda x:self.cross_multihead_attention(x,encoder_ouput,encoder_ouput,src_mask))
        x=self.residual_connections[2](x,self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList,d_model=512):
        super().__init__()
       
        self.layers = layers
        self.norm=nn.LayerNorm(d_model)

    def forward(self,x,encoder_ouput,src_mask,tgt_mask):
        for  layer in self.layers:
            x=layer(x,encoder_ouput,src_mask,tgt_mask)

        return self.norm(x)
    


class ProjectionLayer(nn.Module):
    def __init__(self,d_model:int,vocab_size=int):
        super().__init__()
        self.d_model=d_model
        self.proj=nn.Linear(d_model,vocab_size)
    
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)


class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,
                 src_embd:InputEmbedding,tgt_embd:InputEmbedding,
                 src_pos:PositionalEmbedding,tgt_pos:PositionalEmbedding,
                 projection_layer:ProjectionLayer):
        
        super().__init__()
       
        self.encoder =encoder
        self.decoder=decoder
        self.src_embd=src_embd
        self.tgt_embd=tgt_embd
        self.src_pos=src_pos
        self.tgt_pos=tgt_pos
        self.projection_layer=projection_layer
  

    def encode(self,src,src_mask):
        src=self.src_embd(src)
        src=self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt=self.tgt_embd(tgt)
        tgt=self.tgt_pos(tgt)
        return self.decoder(tgt,encoder_output,src_mask,tgt_mask)
    
    def project(self,x):
        return self.projection_layer(x)



def Build_Transformer(src_vocabsize:int,tgt_vocabsize:int,src_seq_length:int,tgt_seq_length:int,
                      d_model=512,n_layers=6,n_heads=8,dropout=0.1):
    
    
    
    src_embd=InputEmbedding(src_vocabsize,d_model)
    tgt_embd=InputEmbedding(tgt_vocabsize,d_model)
    src_pos=PositionalEmbedding(d_model,src_seq_length,dropout)
    tgt_pos=PositionalEmbedding(d_model,tgt_seq_length,dropout)

    encoder_blocks=[]
    for _ in range(n_layers):
        encoder_self_attention=MultiHeadAttention(d_model,n_heads,dropout)
        ffw=FeedForward(d_model,dropout)
        encoder_block=EncoderBlock(encoder_self_attention,ffw,dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks=[]
    for _ in range(n_layers):
        decoder_self_attention=MultiHeadAttention(d_model,n_heads,dropout)
        decoder_cross_attention=MultiHeadAttention(d_model,n_heads,dropout)
        ffw=FeedForward(d_model,dropout)
        decoder_block=DecoderBlock(decoder_self_attention,decoder_cross_attention,ffw,dropout)
        decoder_blocks.append(decoder_block)


    
    encoder=Encoder(nn.ModuleList(encoder_blocks))
    decoder=Decoder(nn.ModuleList(decoder_blocks))

    projection_layer=ProjectionLayer(d_model,tgt_vocabsize)

    transformer=Transformer(encoder,decoder,src_embd,tgt_embd,src_pos,tgt_pos,projection_layer)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform(p)


    return transformer

    
 
        
        
