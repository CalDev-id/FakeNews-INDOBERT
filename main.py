import os
import sys
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from utils.preprocessor import TwitterDataModule
from models.finetune import FinetuneV1, FinetuneV2
from models.finetune_with_cnn import FinetuneWithCNNv1, FinetuneWithCNNv2
from textwrap import dedent


if __name__ == '__main__':

    seed_everything(seed=42, workers=True)

    parser = argparse.ArgumentParser(description='Trainer Parser')
    parser.add_argument('-m', '--model', choices=['IndoBERT', 'IndoBERTweet', 'IndoRoBERTa_OSCAR', 'IndoRoBERTa_Wiki'], required=True, help='Pretrinaed model choices to train')
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-l', '--max_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('-c', '--cnn', type=bool, default=False, help='CNN Model Type')
    parser.add_argument('-v', '--version', choices=['1', '2'], default=1, help='Model Version')

    args = parser.parse_args()
    config = vars(args)

    # Get arguments values
    model_name = config['model']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    max_length = config['max_length']
    cnn = config['cnn']
    version = int(config['version'])

    print(dedent(f'''
    -----------------------------------
     Finetune Information        
    -----------------------------------
     Name                | Value       
    -----------------------------------
     Model Name          | {model_name}
     Batch Size          | {batch_size}
     Learning Rate       | {learning_rate}
     Input Max Length    | {max_length} 
     Is With CNN         | {cnn} 
     Model Version       | {version} 
    -----------------------------------
    '''))

    pretrained_model_name = {
        'IndoBERT': 'indolem/indobert-base-uncased',
        'IndoBERTweet': 'indolem/indobertweet-base-uncased',
        'IndoRoBERTa_OSCAR': 'flax-community/indonesian-roberta-base',
        'IndoRoBERTa_Wiki': 'cahya/roberta-base-indonesian-522M',
    }

    pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name[model_name], use_fast=False)

    with_cnn_str = ''

    if cnn:
        with_cnn_str = '_CNN'
        if version == 1:
            pretrained_model = AutoModel.from_pretrained(pretrained_model_name[model_name], output_attentions=False, output_hidden_states=True)
            model = FinetuneWithCNNv1(model=pretrained_model, learning_rate=learning_rate)
            data_module = TwitterDataModule(tokenizer=pretrained_tokenizer, max_length=max_length, batch_size=batch_size, recreate=True)
        elif version == 2:
            pretrained_model = AutoModel.from_pretrained(pretrained_model_name[model_name], output_attentions=False, output_hidden_states=True)
            model = FinetuneWithCNNv2(model=pretrained_model, learning_rate=learning_rate)
            data_module = TwitterDataModule(tokenizer=pretrained_tokenizer, max_length=max_length, batch_size=batch_size, recreate=True)
    else:
        if version == 1:
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name[model_name], output_attentions=False, output_hidden_states=False, num_labels=2)
            model = FinetuneV1(model=pretrained_model, learning_rate=learning_rate)
            data_module = TwitterDataModule(tokenizer=pretrained_tokenizer, max_length=max_length, batch_size=batch_size, recreate=True, one_hot_label=True)
        elif version == 2:
            pretrained_model = AutoModel.from_pretrained(pretrained_model_name[model_name], output_attentions=False, output_hidden_states=False)
            model = FinetuneV2(model=pretrained_model, learning_rate=learning_rate)
            data_module = TwitterDataModule(tokenizer=pretrained_tokenizer, max_length=max_length, batch_size=batch_size, recreate=True)

    # Initialize callbacks and progressbar
    tensor_board_logger = TensorBoardLogger('tensorboard_logs', name=f'{model_name}{with_cnn_str}_version{version}/{batch_size}_{learning_rate}')
    csv_logger = CSVLogger('csv_logs', name=f'{model_name}{with_cnn_str}_version{version}/{batch_size}_{learning_rate}')
    checkpoint_callback = ModelCheckpoint(dirpath=f'./checkpoints/{model_name}{with_cnn_str}_version{version}/{batch_size}_{learning_rate}', monitor='val_f1_score', mode='max')
    early_stop_callback = EarlyStopping(monitor='val_f1_score', min_delta=0.00, check_on_train_epoch_end=1, patience=3, mode='max')
    tqdm_progress_bar = TQDMProgressBar()

    # Initialize Trainer
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=50,
        default_root_dir=f'./checkpoints/{model_name}{with_cnn_str}_version{version}/{batch_size}_{learning_rate}',
        callbacks=[checkpoint_callback, early_stop_callback, tqdm_progress_bar],
        logger=[tensor_board_logger, csv_logger],
        log_every_n_steps=5,
        deterministic=True  # To ensure reproducible results
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(datamodule=data_module, ckpt_path='best')