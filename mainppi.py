import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
#from pytorch_lightning.metrics.sklearns import Accuracy
from torchmetrics import Accuracy
from torchmetrics.classification import roc
from torchmetrics import ROC
from transformers import BertTokenizer, BertModel, BertConfig

from torchnlp.encoders import LabelEncoder
from torchnlp.datasets.dataset import Dataset
from torchnlp.utils import collate_tensors
import torchmetrics
import pandas as pd
from test_tube import HyperOptArgumentParser
import os
import re
import requests
from tqdm.auto import tqdm
from datetime import datetime
from collections import OrderedDict
import logging as log
import numpy as np
import glob
import pickle
from Bio import SeqIO
from io import StringIO

class Interaction_dataset():
    """
    Loads the Dataset from the csv files passed to the parser.
    :param hparams: HyperOptArgumentParser obj containg the path to the data files.
    :param train: flag to return the train set.
    :param val: flag to return the validation set.
    :param test: flag to return the test set.
    Returns:
        - Training Dataset, Development Dataset, Testing Dataset
    """


    def collate_lists(self, seq1:list, seq2: list, label: list) -> dict:
        """ Converts each line into a dictionary. """
        collated_dataset = []
        for i in range(len(label)):
            collated_dataset.append({"seq1":str(seq1[i]),"seq2": str(seq2[i]), "label": str(label[i])})
        return collated_dataset

    def load_dataset(self, path):
        file =open(path)
        dictPath = "/data/data/fasta/fasta.p"
        
        with open('/data/data/fasta/dict.p','rb') as pickle_file:
            content_esi = pickle.load(pickle_file)

        with open('/data/data/fasta/fasta.p', 'rb') as pickle_file:
            content_pssm = pickle.load(pickle_file)
        content = {**content_esi, **content_pssm}
        #content = content_esi | content_pssm
        #content.update(content_esi)
        #print('content',content)
        #print(dictFile)
        seq1=[]
        seq2=[]
        labels=[]
        for line in file:
            #print(line)
            #print('toks',line.strip().split(','))
            #print('line is ',line)
            toks = line.strip().split(',')
            #print('prot1',toks[0],'prot2',toks[1],'label',toks[2])
            toks = line.strip().split(',')
            prot1 = str(content[toks[0]])
            prot2 = str(content[toks[1]])
            seq1.append(prot1)
            seq2.append(prot2)
            label = toks[2]
            labels.append(label)
            #print('prot1',prot1,'prot2',prot2,'label',label)

        file.close()
        # Make sure there is a space between every token, and map rarely amino acids
        seq1 = [" ".join("".join(sample.split())) for sample in seq1]
        seq1 = [re.sub(r"[UZOB]", "X", sample) for sample in seq1]
        
        seq2 = [" ".join("".join(sample.split())) for sample in seq2]
        seq2 = [re.sub(r"[UZOB]", "X", sample) for sample in seq2]

        print('path',path,'seq len',len(seq1),'labels len',len(labels))
        assert len(seq1) == len(labels)
        return Dataset(self.collate_lists(seq1,seq2,labels))


class ProtBertPPIClassifier(pl.LightningModule):
    """
    # https://github.com/minimalist-nlp/lightning-text-classification.git

    Sample model to show how to use BERT to classify sentences.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    def __init__(self, hparams) -> None:
        super(ProtBertPPIClassifier, self).__init__()
        self.hparams = hparams
        self.batch_size = self.hparams.batch_size

        self.model_name = "Rostlab/prot_bert_bfd"

        self.dataset = Interaction_dataset()

        self.metric_acc = Accuracy()
        self.test_auroc=torchmetrics.AUROC(num_classes=2)
        # build model
        self.__build_model()
        #self.model.gradient_checkpointing_enable()

        # Loss criterion initialization.
        self.__build_loss()

        if self.hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = self.hparams.nr_frozen_epochs

    def __build_model(self) -> None:
        """ Init BERT model + tokenizer + classification head."""
        print('gradient checkpointing  ',self.hparams.gradient_checkpointing)
        #if args.gradient_checkpointing:
        #self.model.gradient_checkpointing_enable()
        config = BertConfig.from_pretrained("Rostlab/prot_bert_bfd")
        print('Config',config)
        self.ProtBertPPI = BertModel.from_pretrained(self.model_name)
        self.ProtBertPPI.gradient_checkpointing_enable()
        #,gradient_checkpointing=self.hparams.gradient_checkpointing)
        self.encoder_features = 1024

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, do_lower_case=False)

        # Label Encoder
        self.label_encoder = LabelEncoder(
            self.hparams.label_set.split(","), reserved_labels=[]
        )
        self.label_encoder.unknown_index = None

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.encoder_features * 4, self.label_encoder.vocab_size),
            nn.Tanh(),
        )

    def __build_loss(self):
        """ Initializes the loss function/s. """
        weights = [1.0,3.0]
        class_weights = torch.FloatTensor(weights).cuda()
        self._loss = nn.CrossEntropyLoss(weight=class_weights)

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.ProtBertPPI.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.ProtBertPPI.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        """ Predict function.
        :param sample: dictionary with the text we want to classify.
        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

        # https://github.com/UKPLab/sentence-transformers/blob/eb39d0199508149b9d32c1677ee9953a84757ae4/sentence_transformers/models/Pooling.py

    def pool_strategy(self, features,
                      pool_cls=True, pool_max=True, pool_mean=True,
                      pool_mean_sqrt=True):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        if pool_cls:
            output_vectors.append(cls_token)
        if pool_max:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if pool_mean or pool_mean_sqrt:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if pool_mean:
                output_vectors.append(sum_embeddings / sum_mask)
            if pool_mean_sqrt:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        return output_vector

    def forward(self, input_ids1,input_ids2):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]
        Returns:
            Dictionary with model outputs (e.g: logits)
        """

        input_ids1_t = torch.tensor(input_ids1['input_ids'], device=self.device)
        input_ids2_t = torch.tensor(input_ids2['input_ids'], device=self.device)

        attention_mask1 = torch.tensor(input_ids1['attention_mask'],device=self.device)
        attention_mask2 = torch.tensor(input_ids2['attention_mask'],device=self.device)

        word_embeddings1 = self.ProtBertPPI(input_ids1_t,
                                           attention_mask1)[0] 
        word_embeddings2 = self.ProtBertPPI(input_ids2_t,
                                           attention_mask2)[0]
        pooling1 = self.pool_strategy({"token_embeddings": word_embeddings1,
                                      "cls_token_embeddings": word_embeddings1[:, 0],
                                      "attention_mask": attention_mask1,
                                      })
        
        pooling2 = self.pool_strategy({"token_embeddings": word_embeddings2,
                                      "cls_token_embeddings": word_embeddings2[:, 0],
                                      "attention_mask": attention_mask2,
                                      })
        
        #pooling = torch.cat((pooling1,pooling2),1)
        pooling = pooling1*pooling2
        return {"logits": self.classification_head(pooling)}

    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]
        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.

        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)

        inputs1 = self.tokenizer.batch_encode_plus(sample["seq1"],
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.hparams.max_length)

        inputs2 = self.tokenizer.batch_encode_plus(sample["seq2"],
                                                  add_special_tokens=True,
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=self.hparams.max_length)


        if not prepare_target:
            return inputs1,inputs2, {}

        # Prepare target:
        try:
            targets = {"labels": self.label_encoder.batch_encode(sample["label"])}
            return inputs1,inputs2,targets
        except RuntimeError:
            print(sample["label"])
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """
        Runs one training step. This usually consists in the forward function followed
            by the loss function.

        :param batch: The output of your dataloader.
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs1,inputs2,targets = batch
        model_out = self.forward(inputs1,inputs2)
        loss_val = self.loss(model_out, targets)

        tqdm_dict = {"train_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs1,inputs2,targets = batch

        model_out = self.forward(inputs1,inputs2)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = self.metric_acc(labels_hat, y)

        output = OrderedDict({"val_loss": loss_val, "val_acc": val_acc, })

        return output

    def validation_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """

        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        tqdm_dict = {"val_loss": val_loss_mean, "val_acc": val_acc_mean}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "val_loss": val_loss_mean,
        }
        return result

    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.
        Returns:
            - dictionary passed to the validation_end function.
        """
        #print('************TESTING MODE ***************')
        #inputs, targets = batch
        inputs1,inputs2,targets = batch
        model_out = self.forward(inputs1,inputs2)
        loss_test = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = self.metric_acc(labels_hat, y)
        #print('y_hat.cpu()',y_hat,',y.cpu()',y)
        self.test_auroc.update(y_hat,y)
        output = OrderedDict({"test_loss": loss_test, "test_acc": test_acc,"label":y,"output":y_hat })

        return output

    def test_epoch_end(self, outputs: list) -> dict:
        """ Function that takes as input a list of dictionaries returned by the validation_step
        function and measures the model performance accross the entire validation set.

        Returns:
            - Dictionary with metrics to be added to the lightning logger.
        """
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc_mean = torch.stack([x['test_acc'] for x in outputs]).mean()
        #labels = torch.stack([x['label'] for x in outputs])
        #outputs = torch.stack([x['output'] for x in outputs])
        test_auroc = self.test_auroc.compute()
        tqdm_dict = {"test_loss": test_loss_mean, "test_acc": test_acc_mean,"roc":test_auroc}
        result = {
            "progress_bar": tqdm_dict,
            "log": tqdm_dict,
            "test_loss": test_loss_mean,
        }
        return result

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.ProtBertPPI.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    def __retrieve_dataset(self, train=True, val=True, test=True):
        """ Retrieves task specific dataset """
        if train:
            return self.dataset.load_dataset(hparams.train_csv)
        elif val:
            return self.dataset.load_dataset(hparams.dev_csv)
        elif test:
            return self.dataset.load_dataset(hparams.test_csv)
        else:
            print('Incorrect dataset split')

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.__retrieve_dataset(val=False, test=False)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )


    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._dev_dataset = self.__retrieve_dataset(train=False, test=False)
        return DataLoader(
            dataset=self._dev_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._test_dataset = self.__retrieve_dataset(train=False, val=False)
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.prepare_sample,
            num_workers=self.hparams.loader_workers,
        )

    @classmethod
    def add_model_specific_args(
            cls, parser: HyperOptArgumentParser
    ) -> HyperOptArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters.
        :param parser: HyperOptArgumentParser obj
        Returns:
            - updated parser
        """
        parser.opt_list(
            "--max_length",
            default=1536,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=5e-06,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.opt_list(
            "--nr_frozen_epochs",
            default=0,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
            tunable=True,
            options=[0, 1, 2, 3, 4, 5],
        )
        # Data Args:
        parser.add_argument(
            "--label_set",
            default="0,1",
            type=str,
            help="Classification labels set.",
        )
        parser.add_argument(
            "--train_csv",
            default="/data/data/csv/esi/0.train.csv",
            type=str,
            help="Path to the file containing the train data.",
        )
    
        parser.add_argument(
            "--dev_csv",
            default="/data/data/csv/profppi/0.val.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
    
        parser.add_argument(
            "--test_csv",
            default="/data/data/csv/esi/0.test.csv",
            type=str,
            help="Path to the file containing the dev data.",
        )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                       the data will be loaded in the main process.",
        )
        
        parser.add_argument(
            "--gradient_checkpointing",
            default=True,
            type=bool,
            help="Enable or disable gradient checkpointing which use the cpu memory \
                       with the gpu memory to store the model.",
        )
        
        return parser

def setup_testube_logger() -> TestTubeLogger:
    """ Function that sets the TestTubeLogger to be used. """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")

    return TestTubeLogger(
        save_dir="experiments/",
        version=dt_string,
        name="lightning_logs",
    )

logger = setup_testube_logger()

# these are project-wide arguments
parser = HyperOptArgumentParser(
        strategy="random_search",
        description="Minimalist ProtBERT Classifier",
        add_help=True,
)
parser.add_argument("--seed", type=int, default=43, help="Training seed.")
parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
)
# Early Stopping
parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
)
parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
)
parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
parser.add_argument(
        "--min_epochs",
        default=4,
        type=int,
        help="Limits training to a minimum number of epochs",
    )

parser.add_argument(
        "--max_epochs",
        default=4,
        type=int,
        help="Limits training to a max number number of epochs",
    )

# Batching
parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size to be used."
    )
parser.add_argument(
        "--accumulate_grad_batches",
        default=32,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

# gpu/tpu args
parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
parser.add_argument("--tpu_cores", type=int, default=None, help="How many tpus")
parser.add_argument(
        "--val_percent_check",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )

# mixed precision
parser.add_argument("--precision", type=int, default="32", help="full precision or mixed precision mode")
parser.add_argument("--amp_level", type=str, default="O1", help="mixed precision type")

# each LightningModule defines arguments relevant to it
parser = ProtBertPPIClassifier.add_model_specific_args(parser)
hparams = parser.parse_known_args()[0]

"""
Main training routine specific for this project
:param hparams:
"""
seed_everything(hparams.seed)

# ------------------------
# 1 INIT LIGHTNING MODEL
# ------------------------
model = ProtBertPPIClassifier(hparams)

# ------------------------
# 2 INIT EARLY STOPPING
# ------------------------
early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
)

# --------------------------------
# 3 INIT MODEL CHECKPOINT CALLBACK
# -------------------------------
ckpt_path = os.path.join(
        logger.save_dir,
        logger.name,
        f"version_{logger.version}",
        "checkpoints",
)
# initialize Model Checkpoint Saver
checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path + "/" + "{epoch}-{val_loss:.2f}-{val_acc:.2f}",
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        period=1,
        mode=hparams.metric_mode,
    )

# ------------------------
# 4 INIT TRAINER
# ------------------------
trainer = Trainer(
        gpus=hparams.gpus,
        tpu_cores=hparams.tpu_cores,
        logger=logger,
        early_stop_callback=early_stop_callback,
        distributed_backend="dp",
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        val_percent_check=hparams.val_percent_check,
        checkpoint_callback=checkpoint_callback,
        precision=hparams.precision,
        amp_level=hparams.amp_level,
        deterministic=True
    )

# ------------------------
# 6 START TRAINING
# ------------------------
trainer.fit(model)

best_checkpoint_path = glob.glob(ckpt_path + "/*")[0]
print(best_checkpoint_path)

trainer.resume_from_checkpoint = best_checkpoint_path
#trainer.test(model)

model = model.load_from_checkpoint(best_checkpoint_path)

model.eval()
model.freeze()
print('Test_Output---->',trainer.test(model))

'''
sample = {
        "seq": "M S T D T G V S L P S Y E E D Q G S K L I R K A K E A P F V P V G I A G F A A I V A Y G L Y K L K S R G N T K M S I H L I H M R V A A Q G F V V G A M T V G M G Y S M Y R E F W A K P K P",
    }

predictions = model.predict(sample)

print("Sequence Localization Ground Truth is: {} - prediction is: {}".format('Mitochondrion',
                                                                                 predictions['predicted_label']))

sample = {
        "seq": "M R C L P V F I I L L L L I P S A P S V D A Q P T T K D D V P L A S L H D N A K R A L Q M F W N K R D C C P A K L L C C N P",
    }

predictions = model.predict(sample)

print("Sequence Localization Ground Truth is: {} - prediction is: {}".format('Extracellular',
                                                                                 predictions['predicted_label']))

'''
