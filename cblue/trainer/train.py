import os
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

from cblue.utils import seed_everything, ProgressBar, TokenRematch
from cblue.metrics import ee_metric
from cblue.metrics import  ee_commit_prediction
from cblue.models import convert_examples_to_features, save_zen_model


class Trainer(object):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):

        self.args = args
        self.model = model
        self.data_processor = data_processor
        self.tokenizer = tokenizer

        if train_dataset is not None and isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset

        if eval_dataset is not None and isinstance(eval_dataset, Dataset):
            self.eval_dataset = eval_dataset

        self.logger = logger
        self.model_class = model_class
        self.ngram_dict = ngram_dict

    def train(self):
        args = self.args
        logger = self.logger
        model = self.model
        model.to(args.device)

        train_dataloader = self.get_train_dataloader()

        num_training_steps = len(train_dataloader) * args.epochs
        num_warmup_steps = num_training_steps * args.warmup_proportion
        num_examples = len(train_dataloader.dataset)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)
        
        if args.task_name in ['qic', 'qqr', 'qtr', 'sts']:
            seed_everything(args.seed)
            model.zero_grad()

        logger.info("***** Running training *****")
        logger.info("Num samples %d", num_examples)
        logger.info("Num epochs %d", args.epochs)
        logger.info("Num training steps %d", num_training_steps)
        logger.info("Num warmup steps %d", num_warmup_steps)

        global_step = 0
        best_step = None
        best_score = .0
        cnt_patience = 0
        for i in range(args.epochs):
            pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
            for step, item in enumerate(train_dataloader):
                loss = self.training_step(model, item)
                pbar(step, {'loss': loss.item()})

                if args.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                if args.task_name in ['qic', 'qqr', 'qtr', 'sts']:
                    model.zero_grad()
                else:
                    optimizer.zero_grad()

                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    
                    score = self.evaluate(model)
                    if score > best_score:
                        best_score = score
                        best_step = global_step
                        cnt_patience = 0
                        self._save_checkpoint(model, global_step)
                    else:
                        cnt_patience += 1
                        self.logger.info("Earlystopper counter: %s out of %s", cnt_patience, args.earlystop_patience)
                        if cnt_patience >= self.args.earlystop_patience:
                            break
            if cnt_patience >= args.earlystop_patience:
                break

        logger.info("Training Stop! The best step %s: %s", best_step, best_score)
        if args.device == 'cuda':
            torch.cuda.empty_cache()

        self._save_best_checkpoint(best_step=best_step)

        return global_step, best_step

    def evaluate(self, model):
        raise NotImplementedError

    def _save_checkpoint(self, model, step):
        raise NotImplementedError

    def _save_best_checkpoint(self, best_step):
        raise NotImplementedError

    def training_step(self, model, item):
        raise NotImplementedError

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True
        )

    def get_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False
        )

    def get_test_dataloader(self, test_dataset, batch_size=None):
        if not batch_size:
            batch_size = self.args.eval_batch_size
            #batch_size = self.eval_batch_size

        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )


class EETrainer(Trainer):
    def __init__(
            self,
            args,
            model,
            data_processor,
            tokenizer,
            logger,
            model_class,
            train_dataset=None,
            eval_dataset=None,
            ngram_dict=None
    ):
        super(EETrainer, self).__init__(
            args=args,
            model=model,
            data_processor=data_processor,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            logger=logger,
            model_class=model_class,
            ngram_dict=ngram_dict
        )

    def training_step(self, model, item):
        model.train()

        input_ids = item[0].to(self.args.device)
        token_type_ids = item[1].to(self.args.device)
        attention_mask = item[2].to(self.args.device)
        labels = item[3].to(self.args.device)

        if self.args.model_type == 'zen':
            input_ngram_ids = item[4].to(self.args.device)
            ngram_attention_mask = item[5].to(self.args.device)
            ngram_token_type_ids = item[6].to(self.args.device)
            ngram_position_matrix = item[7].to(self.args.device)

        if self.args.model_type == 'zen':
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            labels=labels, ngram_ids=input_ngram_ids, ngram_positions=ngram_position_matrix,
                            ngram_attention_mask=ngram_attention_mask, ngram_token_type_ids=ngram_token_type_ids)
        else:
            outputs = model(labels=labels, input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        loss = outputs[0]
        loss.backward()

        return loss.detach()

    def evaluate(self, model):
        args = self.args
        logger = self.logger
        eval_dataloader = self.get_eval_dataloader()
        num_examples = len(eval_dataloader.dataset)

        preds = None
        eval_labels = None

        logger.info("***** Running evaluation *****")
        logger.info("Num samples %d", num_examples)
        for step, item in enumerate(eval_dataloader):
            model.eval()

            input_ids = item[0].to(self.args.device)
            token_type_ids = item[1].to(self.args.device)
            attention_mask = item[2].to(self.args.device)
            labels = item[3].to(self.args.device)

            if args.model_type == 'zen':
                input_ngram_ids = item[4].to(self.args.device)
                ngram_attention_mask = item[5].to(self.args.device)
                ngram_token_type_ids = item[6].to(self.args.device)
                ngram_position_matrix = item[7].to(self.args.device)

            with torch.no_grad():
                if self.args.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    labels=labels, ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(labels=labels, input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                # outputs = model(labels=labels, **inputs)
                loss, logits = outputs[:2]
                # active_index = inputs['attention_mask'].view(-1) == 1
                active_index = attention_mask.view(-1) == 1
                active_labels = labels.view(-1)[active_index]
                logits = logits.argmax(dim=-1)
                active_logits = logits.view(-1)[active_index]

            if preds is None:
                preds = active_logits.detach().cpu().numpy()
                eval_labels = active_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, active_logits.detach().cpu().numpy(), axis=0)
                eval_labels = np.append(eval_labels, active_labels.detach().cpu().numpy(), axis=0)

        p, r, f1, _ = ee_metric(preds, eval_labels)
        logger.info("%s-%s precision: %s - recall: %s - f1 score: %s", args.task_name, args.model_name, p, r, f1)
        return f1

    def predict(self, model, test_dataset):
        args = self.args
        logger = self.logger
        test_dataloader = self.get_test_dataloader(test_dataset)
        num_examples = len(test_dataloader.dataset)
        model.to(args.device)

        predictions = []

        logger.info("***** Running prediction *****")
        logger.info("Num samples %d", num_examples)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Prediction')
        for step, item in enumerate(test_dataloader):
            model.eval()

            input_ids = item[0].to(self.args.device)
            token_type_ids = item[1].to(self.args.device)
            attention_mask = item[2].to(self.args.device)

            if args.model_type == 'zen':
                input_ngram_ids = item[3].to(self.args.device)
                ngram_attention_mask = item[4].to(self.args.device)
                ngram_token_type_ids = item[5].to(self.args.device)
                ngram_position_matrix = item[6].to(self.args.device)

            with torch.no_grad():
                if self.args.model_type == 'zen':
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                    ngram_ids=input_ngram_ids,
                                    ngram_positions=ngram_position_matrix,
                                    ngram_token_type_ids=ngram_token_type_ids,
                                    ngram_attention_mask=ngram_attention_mask)
                else:
                    outputs = model(input_ids=input_ids, token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

                if args.model_type == 'zen':
                    logits = outputs.detach()
                else:
                    logits = outputs[0].detach()
                # active_index = (inputs['attention_mask'] == 1).cpu()
                active_index = attention_mask == 1
                preds = logits.argmax(dim=-1).cpu()

                for i in range(len(active_index)):
                    predictions.append(preds[i][active_index[i]].tolist())
            pbar(step=step, info="")

        # test_inputs = [list(text) for text in test_dataset.texts]
        test_inputs = test_dataset.texts
        predictions = [pred[1:-1] for pred in predictions]
        predicts = self.data_processor.extract_result(predictions, test_inputs)
        ee_commit_prediction(dataset=test_dataset, preds=predicts, output_dir=args.result_output_dir)

    def _save_checkpoint(self, model, step):
        output_dir = os.path.join(self.args.output_dir, 'checkpoint-{}'.format(step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.args.model_type == 'zen':
            save_zen_model(output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=output_dir)
        self.logger.info('Saving models checkpoint to %s', output_dir)

    def _save_best_checkpoint(self, best_step):
        model = self.model_class.from_pretrained(os.path.join(self.args.output_dir, f'checkpoint-{best_step}'),
                                                 num_labels=self.data_processor.num_labels)
        if self.args.model_type == 'zen':
            save_zen_model(self.args.output_dir, model=model, tokenizer=self.tokenizer,
                           ngram_dict=self.ngram_dict, args=self.args)
        else:
            model.save_pretrained(self.args.output_dir)
            torch.save(self.args, os.path.join(self.args.output_dir, 'training_args.bin'))
            self.tokenizer.save_vocabulary(save_directory=self.args.output_dir)
        self.logger.info('Saving models checkpoint to %s', self.args.output_dir)





    
    