# author: sunshine
# datetime:2021/7/2 下午2:06


import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class NewsClassifier(nn.Module):
    def __init__(self, args, num_label):
        super(NewsClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(args.bert_path)
        self.fc = torch.nn.Linear(768, num_label)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x = self.bert(input_ids, attention_mask, token_type_ids)
        x = x[0][:, 0, :]  # 取cls向量
        x = self.fc(x)
        return x


class Trainer(object):

    def __init__(self, args, data_loaders, tokenizer, num_labels):

        self.args = args
        self.num_labels = num_labels

        self.tokenizer = tokenizer
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")

        self.model = NewsClassifier(args, num_labels)

        self.model.to(self.device)
        if args.train_mode == "eval":
            self.resume()

        self.train_dataloader, self.dev_dataloader = data_loaders

        # 设置优化器，优化策略
        train_steps = (len(self.train_dataloader) / args.batch_size) * args.epoch_num
        self.optimizer, self.schedule = self.set_optimizer(args=args,
                                                           model=self.model,
                                                           train_steps=train_steps)

        self.ce = torch.nn.CrossEntropyLoss()
        self.kld = torch.nn.KLDivLoss(reduction="none")

    def loss_fnc(self, y_pred, y_true, alpha=4):
        """配合R-Drop的交叉熵损失
            """

        loss1 = self.ce(y_pred, y_true)
        loss2 = self.kld(torch.log_softmax(y_pred[::2], dim=1), y_pred[1::2].softmax(dim=-1)) + \
                self.kld(torch.log_softmax(y_pred[1::2], dim=1), y_pred[::2].softmax(dim=-1))

        return loss1 + torch.mean(loss2) / 4 * alpha

    def set_optimizer(self, args, model, train_steps=None):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # optimizer, num_warmup_steps, num_training_steps
        schedule = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=train_steps
        )
        return optimizer, schedule

    def train(self, args):

        best_f1 = 0.0
        self.model.train()
        step_gap = 10
        step_eval = 500
        for epoch in range(int(args.epoch_num)):

            for step, batch in tqdm(enumerate(self.train_dataloader)):

                loss = self.forward(batch, is_eval=False)
                if step % step_gap == 0:
                    print(u"step {} / {} of epoch {}, train/loss: {}".format(step,
                                                                             len(self.train_dataloader) / args.batch_size,
                                                                             epoch, loss.item()))

                if step % step_eval == 0:

                    acc = self.evaluate(self.dev_dataloader)
                    print("acc: {}".format(acc))
                    if acc >= best_f1:
                        best_f1 = acc

                        # 保存模型
                        self.save()

    def forward(self, batch, is_eval=False):
        batch = tuple(t.to(self.device) for t in batch)
        if not is_eval:
            input_ids, attention_mask, token_type_ids, label = batch
            self.optimizer.zero_grad()
            span_logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            loss = self.loss_fnc(y_pred=span_logits, y_true=label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.warmup_proportion)
            # loss = loss.item()
            self.optimizer.step()
            self.schedule.step()

            return loss
        else:
            input_ids, attention_mask, token_type_ids, label = batch
            span_logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            y_pred = torch.argmax(span_logits, dim=-1)
            y_true = label
            tmp_total = len(y_true)
            tmp_right = (y_true == y_pred).sum().cpu().numpy()
            return tmp_total, tmp_right

    def resume(self):
        resume_model_file = self.args.output + "/pytorch_model.bin"
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def save(self):
        logger.info("** ** * Saving fine-tuned model ** ** * ")
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self
        output_model_file = self.args.output + "/pytorch_model.bin"
        torch.save(model_to_save.state_dict(), str(output_model_file))

    def evaluate(self, dataloader):
        """验证
        """
        self.model.eval()
        total, right = 0.0, 0.0
        with torch.no_grad():
            for batch in dataloader:
                tmp_total, tmp_right = self.forward(batch=batch, is_eval=True)
                total += tmp_total
                tmp_right += tmp_right
        self.model.train()
        return right / total


if __name__ == '__main__':
    KL_criterion = torch.nn.KLDivLoss(size_average=False)
    a = torch.tensor([0.2, 0.1, 0.3, 0.4])
    b = torch.tensor([0.1, 0.2, 0.3, 0.4])

    loss = F.kl_div(a.log(), b)
    print(loss)
