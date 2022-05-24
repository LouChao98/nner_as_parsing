import logging
import os
import pickle
import re
from functools import partial

import pytorch_lightning as pl
from fastNLP import DataSetIter
from omegaconf import OmegaConf

import src
from src.data.sampler import BasicSampler, ConstantTokenNumSampler
from src.my_typing import *

log = logging.getLogger('datamodule')


class DataModule(pl.LightningDataModule):
    INPUTS = ('id', 'words', 'seq_len')
    TARGETS = ('target',)
    EXTRA_VOCAB = ()  # fields need to build extra_vocab
    LOADER = None

    def __init__(self,
                 loader,
                 normalize_word=True,
                 build_no_create_entry=True,
                 build_word_for_others=False,
                 max_len=None,
                 distributed=False,
                 suffix=None):
        super(DataModule, self).__init__()
        self._loader = loader
        self.loader = None  # this field will be set automatically.
        self.normalize_word = normalize_word
        self.build_no_create_entry = build_no_create_entry
        self.build_word_for_others = build_word_for_others  # build word vocab for unlabeled, augumented ...
        self.max_len = max_len or {}  # a dict, {name: len}
        self.distributed = distributed

        self.datasets: Dict[str, DataSet] = {}
        self.vocabs: Dict[str, Vocabulary] = {}

        if 'MANUAL' not in src.g_cfg:
            cfg = src.g_cfg
            self.watch_loss = 'loss' in cfg.watch_field
            self.test_when_val = cfg.runner.test_when_val
            self.suffix = ['']
            if self.normalize_word:
                self.suffix.append('nw')
            if self.build_no_create_entry:
                self.suffix.append('nc')
            if self.build_word_for_others:
                self.suffix.append('wo')
            if suffix:
                self.suffix.extend(suffix)
            self.suffix = '_'.join(self.suffix)
        else:
            self.watch_loss, self.test_when_val, self.suffix = False, False, ''

    # ========================================================================
    # Override these functions to customize.

    def _load(self, path, is_eval, set_target) -> DataSet:
        raise NotImplementedError

    def _post_init_vocab(self, datasets: Dict[str, DataSet]):
        pass  # after init_vocab, before apply_vocab

    def _post_apply_vocab(self, datasets: Dict[str, DataSet]):
        pass

    def get_bos_eos_adder(self, field):
        if field in ('id', 'seq_len'): return None
        return lambda x: ['<bos>'] + x + ['<eos>']

    # ========================================================================

    def load(self, path, is_eval=False, set_target=True, max_len=None, loader=None, name=None):
        """
        add id, copy raw_words, backup inputs.
        <bos>/<eos> for INPUTS will be handled automatically.
        <bos>/<eos> for TARGETS should be handled manually.
        set INPUTS/TARGETS is moved to setup().
        """
        self.loader = loader if loader is not None else self._loader
        ds = self._load(path, is_eval, set_target)

        # backup input fields
        for field in self.INPUTS:
            if field in ('id', 'words', 'seq_len'): continue
            ds.copy_field(field, f'raw_{field}')

        # process words
        if 'words' not in ds:
            ds.copy_field('raw_words', 'words')
            if self.normalize_word:
                log.debug('Normalizing words.')
                ds.apply_field(self.normalize_word_func, 'words', 'words')
        elif self.normalize_word:
            log.warning("normalize_word is skipped because 'words' exists.")

        # add <bos> and <eos>
        for field in self.INPUTS:
            if (f := self.get_bos_eos_adder(field)) is not None:
                ds.apply_field(f, field, field)

        if set_target:
            for field in self.TARGETS:
                if (f := self.get_bos_eos_adder(field)) is not None:
                    ds.apply_field(f, field, field)

        # in apply_max_len
        if 'id' not in ds:
            ds.add_field('id', list(range(len(ds))), padder=None)
        else:
            log.warning('"id" is created before the default pipeline. Make sure the padder is set correctly.')

        if 'seq_len' not in ds:
            ds.add_seq_len('words')
        else:
            log.warning('"seq_len" is created before the default pipeline. Make sure the padder is set correctly.')

        return ds

    def get_create_entry_ds(self):
        create_entry_ds = [ds for name, ds in self.datasets.items() if name not in ('dev', 'test')] \
            if self.build_word_for_others else [self.datasets['train']]
        create_entry_ds = list(filter(lambda x: isinstance(x, DataSet), create_entry_ds))
        assert len(create_entry_ds) > 0, 'no create entry ds available'

        # create_entry_ds = [d.drop(lambda x: x['empty'], inplace=False) for d in create_entry_ds]
        return create_entry_ds

    def get_no_create_entry_ds(self):
        if self.build_no_create_entry:
            no_create_entry_ds = [self.datasets['dev'], self.datasets['test']]
            no_create_entry_ds = list(filter(lambda x: isinstance(x, DataSet), no_create_entry_ds))
        else:
            no_create_entry_ds = []

        # no_create_entry_ds = [d.drop(lambda x: x['empty'], inplace=False) for d in no_create_entry_ds]
        return no_create_entry_ds

    def init_vocab(self):
        # set self.vocabs[XXX] = None to skip auto init.

        # init vocab
        if 'words' not in self.vocabs:
            self.vocabs['words'] = Vocabulary(specials=['<bos>', '<eos>'])
        else:
            assert self.vocabs['words'] is None, 'Must be None to skip auto init'
        for field in self.EXTRA_VOCAB:
            if field in self.vocabs:
                assert self.vocabs[field] is None, 'Must be None to skip auto init'
                continue
            if field in self.INPUTS:
                self.vocabs[field] = Vocabulary(specials=['<bos>', '<eos>'])
            else:
                self.vocabs[field] = Vocabulary(padding=None, unknown='<unk>')

        # load from text
        if src.g_cfg.data.vocab is not None:
            for key, path in src.g_cfg.data.vocab.items():
                log.debug(f'Loading vocab {key} from {path}.')
                self.vocabs[key] = Vocabulary.load(path)

        create_entry_ds = self.get_create_entry_ds()
        no_create_entry_ds = self.get_no_create_entry_ds()

        # auto build
        if self.vocabs['words'] is not None:
            self.vocabs['words'].from_dataset(*create_entry_ds,
                                              field_name='words',
                                              no_create_entry_dataset=no_create_entry_ds)
        for field in self.EXTRA_VOCAB:
            if self.vocabs[field] is not None:
                self.vocabs[field].from_dataset(self.datasets['train'], field_name=field)

        self._post_init_vocab(self.datasets)
        self._check_all_vocab_initialized()
        self.apply_vocab()
        self._post_apply_vocab(self.datasets)

    def _check_all_vocab_initialized(self):
        for name, vocab in self.vocabs.items():
            if vocab is None:
                raise ValueError(f'Vocab {name} is set to manual setup, but not.')

    def apply_vocab(self, ds=None):
        if ds is None:
            to_be_indexed = self.datasets.values()
        elif isinstance(ds, (list, tuple)):
            to_be_indexed = ds
        else:
            to_be_indexed = [ds]
        for ds in to_be_indexed:
            if not isinstance(ds, DataSet):
                continue
            for field, vocab in self.vocabs.items():
                if field in ds:
                    vocab.index_dataset(ds, field_name=field)

    def apply_max_len(self):
        for name, ds in self.datasets.items():
            if (max_len := self.max_len.get(name)) is not None:
                ds.drop(lambda i: i['seq_len'] > max_len)
                ds.add_field('id', list(range(len(ds))), padder=None)
                # TODO detect user defined id

    def setup(self, stage=None):
        can_load, need_save = False, False

        if src.g_cfg.data.cache and src.g_cfg.data.use_cache:
            cache_path = src.g_cfg.data.cache + self.suffix
            if os.path.exists(cache_path):
                can_load = True
            else:
                folder, filename = os.path.split(cache_path)
                os.makedirs(folder, exist_ok=True)
                need_save = True

        if can_load:
            log.info(f'Loading cached data from {cache_path}.')
            with open(cache_path, 'rb') as f:
                d = pickle.load(f)
                self.datasets = d['datasets']
                self.vocabs = d['vocabs']
        else:
            log.info('Loading data.')
            data_cfg = src.g_cfg.data
            self.datasets['train'] = self.load(data_cfg.train, name='train')
            self.datasets['test'] = self.load(data_cfg.test, is_eval=True, name='test')
            self.datasets['dev'] = self.load(data_cfg.dev, is_eval=True, name='dev')


            self.init_vocab()

        if need_save:
            log.info(f'Saving data to {cache_path}')
            with open(cache_path, 'wb') as f:
                pickle.dump({'datasets': self.datasets, 'vocabs': self.vocabs}, f)

        self.apply_max_len()

        # This should match the settings when 'can_load=False'
        for ds in self.datasets.values():
            ds.set_input(*self.INPUTS)
        for name, ds in self.datasets.items():
            if name in ('train', 'dev', 'test'):
                ds.set_target(*self.TARGETS)

        for name, ds in self.datasets.items():
            log.info(f'{name} contains {len(ds)} instances and {sum(ds["seq_len"].content) - 2 * len(ds)} tokens.')
        return self

    def dataloader(self, name):
        cfg = {**src.g_cfg.dataloader.default, **src.g_cfg.dataloader.get(name, {})}
        if name in ('dev', 'test'):
            cfg['no_drop'] = True
            cfg['shuffle'] = False
        i = get_dataset_iter(self.datasets[name], **cfg)
        log.info(f'Getting dataloader={name}, size={len(i) if hasattr(i, "__len__") else "unavailable"}')
        return i

    def train_dataloader(self):
        loaders = {'train': self.dataloader('train')}
        for key in self.datasets:
            if key in ('train', 'dev', 'test'):
                continue
            loaders[key] = self.dataloader(key)
        log.info(f'Returning {len(loaders)} loader(s) as train_dataloader.')
        return loaders

    def val_dataloader(self):
        if self.test_when_val:
            return self.dataloader('dev'), self.dataloader('test')
        return self.dataloader('dev')

    def test_dataloader(self):
        return self.dataloader('test')

    def normalize_chars(self, w: str):
        if w == '-LRB-':
            return '('
        elif w == '-RRB-':
            return ')'
        elif w == '-LCB-':
            return '{'
        elif w == '-RCB-':
            return '}'
        elif w == '-LSB-':
            return '['
        elif w == '-RSB-':
            return ']'
        return w.replace(r'\/', '/').replace(r'\*', '*')

    def normalize_one_word_func(self, w):
        return re.sub(r'\d', '0', self.normalize_chars(w))

    def normalize_word_func(self, ws: List[str]):
        return [re.sub(r'\d', '0', self.normalize_chars(w)) for w in ws]

    def get_vocab_count(self):
        return OmegaConf.create({f'n_{name}': len(vocab) for name, vocab in self.vocabs.items()})

def get_dataset_iter(ds: DataSet,
                     token_size,
                     num_bucket,
                     batch_size=-1,
                     single_sent_threshold=-1,
                     shuffle=True,
                     no_drop=False,
                     fully_shuffle=False,
                     sort_in_batch=True,
                     force_same_len=False,
                     weight=None,
                     **kwargs):
    kwargs.setdefault('num_workers', 4)
    kwargs.setdefault('pin_memory', True)
    num_accumulate = src.g_cfg.trainer.accumulate_grad_batches
    # kwargs['num_workers'] = 0
    if weight is not None:
        weight = ds[weight].content

    if num_bucket > 1:
        get_sampler = partial(ConstantTokenNumSampler,
                              max_token=token_size // num_accumulate,
                              max_sentence=batch_size,
                              num_bucket=num_bucket,
                              single_sent_threshold=single_sent_threshold,
                              sort_in_batch=sort_in_batch,
                              shuffle=shuffle,
                              no_drop=no_drop,
                              fully_shuffle=shuffle if fully_shuffle is None else (fully_shuffle and shuffle),
                              force_same_len=force_same_len,
                              weight=weight)
    else:
        assert batch_size > 0
        assert weight is None
        get_sampler = partial(BasicSampler,
                              batch_size=batch_size // num_accumulate,
                              sort_in_batch=sort_in_batch,
                              shuffle=shuffle)
    # kwargs['pin_memory'] = False
    if isinstance(ds, DataSet):
        return DataSetIter(ds, batch_sampler=get_sampler([l - 1 for l in ds['seq_len'].content]), **kwargs)
    raise ValueError
