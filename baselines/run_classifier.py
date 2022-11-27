import os
import sys
sys.path.append('.')
import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AlbertForSequenceClassification, \
    BertForTokenClassification, AlbertForTokenClassification

from cblue.data import EEDataset, EEDataProcessor
from cblue.trainer import EETrainer
from cblue.utils import init_logger, seed_everything
from cblue.models import ZenConfig, ZenNgramDict, ZenForSequenceClassification, ZenForTokenClassification


TASK_DATASET_CLASS = {
    'ee': (EEDataset, EEDataProcessor)
}

TASK_TRAINER = {
    'ee': EETrainer
}

MODEL_CLASS = {
    'bert': (BertTokenizer, BertForSequenceClassification),
    'roberta': (BertTokenizer, BertForSequenceClassification),
    'albert': (BertTokenizer, AlbertForSequenceClassification),
    'zen': (BertTokenizer, ZenForSequenceClassification)
}

TOKEN_MODEL_CLASS = {
    'bert': (BertTokenizer, BertForTokenClassification),
    'roberta': (BertTokenizer, BertForTokenClassification),
    'albert': (BertTokenizer, AlbertForTokenClassification),
    'zen': (BertTokenizer, ZenForTokenClassification)
}


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir',default='CBLUEDatasets')
    parser.add_argument('--model_dir',default="data/model_data")
    parser.add_argument('--model_type',default="bert")
    parser.add_argument('--model_name',default='chinese-bert-wwm-ext')
    parser.add_argument('--output_dir',default="data/output/ee/chinese-bert-wwm-ext")
    parser.add_argument('--result_output_dir',default="data/result_output")
    parser.add_argument('--task_name',default="ee")

    parser.add_argument('--max_length',default=128)
    parser.add_argument('--train_batch_size',default=16)
    parser.add_argument('--eval_batch_size',default=32)
    parser.add_argument('--learning_rate',default=3e-5)
    parser.add_argument('--weight_decay',default=0.01)
    parser.add_argument('--adam_epsilon',default=1e-8)
    parser.add_argument('--max_grad_norm',default=1.0)
    parser.add_argument('--epochs',default=5)
    parser.add_argument('--warmup_proportion',default=0.1)
    parser.add_argument('--earlystop_patience',default=100)
    parser.add_argument('--logging_steps',default=200)
    parser.add_argument('--save_steps',default=200)
    parser.add_argument('--seed',default=2021)

    args = parser.parse_args()

    do_train=True

    logger = init_logger(os.path.join(args.output_dir, f'{args.task_name}_{args.model_name}.log'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    seed_everything(args.seed)

    if 'albert' in args.model_name:
        args.model_type = 'albert'

    tokenizer_class, model_class = MODEL_CLASS[args.model_type]
    dataset_class, data_processor_class = TASK_DATASET_CLASS[args.task_name]
    trainer_class = TASK_TRAINER[args.task_name]
    print("训练测试集：",trainer_class)

    if args.task_name == 'ee':
        tokenizer_class, model_class = TOKEN_MODEL_CLASS[args.model_type]

    #logger.info("Training/evaluation parameters %s", args)
    if do_train:
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.model_dir, args.model_name))

        # compatible with 'ZEN' model
        ngram_dict = None
        if args.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(args.model_dir, args.model_name), tokenizer=tokenizer)

        data_processor = data_processor_class(root=args.data_dir)
        train_samples = data_processor.get_train_sample()
        eval_samples = data_processor.get_dev_sample()

        
        train_dataset = dataset_class(train_samples, data_processor, tokenizer, mode='train',
                                          model_type=args.model_type, ngram_dict=ngram_dict, max_length=args.max_length)
        eval_dataset = dataset_class(eval_samples, data_processor, tokenizer, mode='eval',
                                         model_type=args.model_type, ngram_dict=ngram_dict, max_length=args.max_length)


        model = model_class.from_pretrained(os.path.join(args.model_dir, args.model_name),
                                            num_labels=data_processor.num_labels)

        trainer = trainer_class(args=args, model=model, data_processor=data_processor,
                                tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset,
                                logger=logger, model_class=model_class, ngram_dict=ngram_dict)

        global_step, best_step = trainer.train()

    else:    
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)

        ngram_dict = None
        if args.model_type == 'zen':
            ngram_dict = ZenNgramDict(os.path.join(args.model_dir, args.model_name), tokenizer=tokenizer)
        
        data_processor = data_processor_class(root=args.data_dir)
        test_samples = data_processor.get_test_sample()

        
        test_dataset = dataset_class(test_samples, data_processor, tokenizer, mode='test', ngram_dict=ngram_dict,
                                         max_length=args.max_length, model_type=args.model_type)
        
            
        model = model_class.from_pretrained(args.output_dir, num_labels=data_processor.num_labels)
        trainer = trainer_class(args=args, model=model, data_processor=data_processor,
                                tokenizer=tokenizer, logger=logger, model_class=model_class, ngram_dict=ngram_dict)
        trainer.predict(test_dataset=test_dataset, model=model)


if __name__ == '__main__':
    main()
