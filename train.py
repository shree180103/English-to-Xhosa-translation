import torch
from torch import nn
from datasets import load_dataset,concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset,DataLoader,random_split
from pathlib import Path
import dataset 
from dataset import BillingualDataset,causal_mask 
import model 
from model import Build_Transformer
from  torch.utils.tensorboard import SummaryWriter 
import config
from config import get_weights_file_path,get_config,latest_weights_file_path
from tqdm import tqdm
import torchmetrics
import warnings
import re
import datasets
from datasets import Dataset, DatasetDict
from ast import literal_eval
import sentencepiece as spm
import tempfile



def greedy_decode(model,source,source_mask,tokenizer_src,tokenizer_tgt,max_len,device):
    sos_idx=tokenizer_tgt.piece_to_id('[SOS]')
    eos_idx=tokenizer_tgt.piece_to_id('[EOS]')

    #precomute encoder output and resuse for every input token we gent from the decoder
    encoder_output=model.encode(source,source_mask)
    #initialize the decoder with the soso token
    decoder_input=torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1)==max_len:
            break

        decoder_mask=causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculating the output 
       
        out=model.decode(encoder_output,source_mask,decoder_input,decoder_mask)

        # Get max token
        prob=model.project(out[:,-1])        
        _,next_word=torch.max(prob,dim=1)
        decoder_input=torch.cat([decoder_input,torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],dim=1)

        if next_word==eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model,validation_ds,tokenizer_src,tokenizer_tgt,max_len,device,print_msg,global_state,
                   writer,num_examples=2):
    model.eval()
    count=0

    source_texts=[]
    expected=[]
    predicted=[]

    console_width=80

    with torch.no_grad():
        for batch in validation_ds:
            count+=1
            encoder_input=batch['encoder_input'].to(device) #(B,Tx)
            encoder_mask=batch['encoder_mask'].to(device)   #(B,1,1,T)

            assert encoder_input.size(0)==1, "Batch size must be 1 for validation"
           
            model_out=greedy_decode(model,encoder_input,encoder_mask,tokenizer_src,tokenizer_tgt,max_len,device)

            source_text=batch['src_text'][0]
            target_text=batch['tgt_text'][0]

            model_out_text=tokenizer_tgt.decode(model_out.detach().cpu().numpy().tolist())
            model_out_text = model_out_text.replace('[SOS]', '').replace('[EOS]', '').strip()

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
           

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)

            print_msg('-'*console_width)
            print_msg(f'SOURCE:{source_text}')
            print_msg(f'TARGET:{target_text}')
            print_msg(f'PREDICTED:{model_out_text}')

            print_msg(f'Blue score:{bleu}, char err rate:{cer}, word err rate:{wer}')
 

            if count==num_examples:
                break




# def get_all_sentances(ds,lang):
#     for item in ds:
#         yield item['translation'][lang]


# def get_or_build_tokenizer(config,ds,lang):
#     tokenizer_path=Path(config['tokenizer_file'].format(lang))
#     if not Path.exists(tokenizer_path):
#         tokenizer=Tokenizer(BPE(unk_token='[UNK]'))
#         tokenizer.pre_tokenizer=Whitespace()
#         trainer=BpeTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
#         tokenizer.train_from_iterator(get_all_sentances(ds,lang),trainer=trainer)
#         tokenizer.save(str(tokenizer_path))

#     else:
#         tokenizer=Tokenizer.from_file(str(tokenizer_path))

#     return tokenizer

import sentencepiece as spm
import tempfile
from pathlib import Path

def get_all_sentances(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # Assume config['tokenizer_file'] is a format string, e.g., "tokenizer_{}.model"
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    
    if not tokenizer_path.exists():
        # Write sentences to a temporary file for training
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp_file:
            for sentence in get_all_sentances(ds, lang):
                tmp_file.write(sentence.strip() + "\n")
            temp_file_path = tmp_file.name

        # Set model_prefix to be the tokenizer path without the file extension.
        model_prefix = str(tokenizer_path.with_suffix(''))  # e.g., "tokenizer_en" if tokenizer_path is "tokenizer_en.model"

        # Train the SentencePiece model.
        # Adjust vocab_size as needed. The character_coverage=1.0 ensures full character inclusion (e.g., for Xhosa click sounds).
        spm.SentencePieceTrainer.train(
            input=temp_file_path,
            model_prefix=model_prefix,
            vocab_size=config.get('vocab_size', 32000),
            character_coverage=1.0,
            model_type='bpe',  # You can choose 'unigram' or other types if preferred.
            user_defined_symbols=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        )

        # Clean up the temporary training file.
        Path(temp_file_path).unlink()

        # Load the newly trained SentencePiece model.
        tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    else:
        # Load the existing SentencePiece model.
        tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

    return tokenizer



def get_ds(config):
    # ds_raw_1=load_dataset('jonathansuru/en_xho',split='train')
    # ds_raw_2=load_dataset('jonathansuru/en_xho',split='validation')
    # ds_raw_3=load_dataset('jonathansuru/en_xho',split='test')
    # ds_raw_hug1=concatenate_datasets([ds_raw_1, ds_raw_2,ds_raw_3])
   
    with open('translation_corpus (3).txt', 'r', encoding='utf-8') as f:
        content = f.read()

# Parse the content into a Python dictionary
    data = literal_eval(content)
    parsed_data = []
    for value in data.values():
        match = re.search(
            r"<en>(?P<en>.*?)<en>\s*:\s*<xh>(?P<xh>.*?)<xh>",
            value.strip(),
            re.DOTALL
        )
        if match:
            en_text = match.group("en").strip()
            xh_text = match.group("xh").strip()
            parsed_data.append({"en": en_text, "xh": xh_text})

    ds_raw= Dataset.from_dict({"translation": parsed_data})
    # ds_raw_hug2 = ds_raw_hug2.cast(ds_raw_hug1.features)

    # ds_raw=concatenate_datasets([ds_raw_hug1,ds_raw_hug2])
    ds_raw=[item for item in ds_raw if ((len(item['translation'][config['lang_src']])<=300) and (len(item['translation'][config['lang_tgt']])<=300))]
    print(len(ds_raw))


    #build tokenizer 
    tokenizer_src=get_or_build_tokenizer(config,ds_raw,config['lang_src'])
    tokenizer_tgt=get_or_build_tokenizer(config,ds_raw,config['lang_tgt'])

    train_ds_size=int(0.9*len(ds_raw))
    val_ds_size=len(ds_raw)-train_ds_size
    train_ds_raw,val_ds_raw=random_split(ds_raw,[train_ds_size,val_ds_size])

    train_ds=BillingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])
    val_ds=BillingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'])

    max_len_src=0
    max_len_tgt=0
    src_len=[]
    tgt_len=[]
    for item in ds_raw:
        src_ids=tokenizer_src.encode(item['translation'][config['lang_src']],out_type=int)
        tgt_ids=tokenizer_tgt.encode(item['translation'][config['lang_tgt']],out_type=int)

        max_len_src=max(max_len_src,len(src_ids))
        max_len_tgt=max(max_len_tgt,len(tgt_ids))

       
    print(f'Max length of src sentance:{max_len_src}')
    print(f'Max length of tgt sentance:{max_len_tgt}')




    

    train_dataloader=DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,

    )
    
    val_dataloader=DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,

    )

    return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt



def get_model(config,vocab_src_len,vocab_tgt_len):
    model=Build_Transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],d_model=config['d_model'])
    
    return model

loss_list=[]
def train_model(config):

    device="cuda" if torch.cuda.is_available() else "cpu"
    # device='cpu'
    print(f'using device {device}')

   
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt=get_ds(config)

    model = get_model(config, tokenizer_src.get_piece_size(), tokenizer_tgt.get_piece_size()).to(device)

    #tensorboad
    writer=SummaryWriter(config['experiment_name'])

    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'],eps=1e-9)

    initial_epoch=0
    global_step=0

    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    
    loss_fn=nn.CrossEntropyLoss(ignore_index=tokenizer_src.piece_to_id('[PAD]'),label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch,config['num_epochs']):
        
        batch_iterator=tqdm(train_dataloader,desc=f'processing epoch{epoch:02d}')
        for batch in batch_iterator:
            model.train()
            encoder_input=batch['encoder_input'].to(device) #(B,Tx)
            decoder_input=batch['decoder_input'].to(device) #(B,Ty)
            encoder_mask=batch['encoder_mask'].to(device)   #(B,1,1,T)
            decoder_mask=batch['decoder_mask'].to(device)   #(B,1,T,T)

            enocder_output=model.encode(encoder_input,encoder_mask) #(B,T,d_model)
            decoder_output=model.decode(enocder_output,encoder_mask,decoder_input,decoder_mask) #(B,T,d_model)
            proj_output=model.project(decoder_output) #(B,T,tgt_vocab_size)

            label=batch['label'].to(device) #(B,T)
            loss=loss_fn(proj_output.view(-1,tokenizer_tgt.get_piece_size()),label.view(-1))
            batch_iterator.set_postfix({f'loss':f"{loss.item():6.3f}"})

            writer.add_scalar('train loss',loss.item(),global_step)
            writer.flush()   
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config['seq_len'],device,
            #                lambda msg:batch_iterator.write(msg),global_step,writer)
            
            
            
            global_step+=1

        run_validation(model,val_dataloader,tokenizer_src,tokenizer_tgt,config['seq_len'],device,
                           lambda msg:batch_iterator.write(msg),global_step,writer)
    

        #save model
            
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
            

if __name__ =='__main__':
    warnings.filterwarnings('ignore')
    config=get_config()
    train_model(config)