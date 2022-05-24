If you want to change the default config of dataloader, please refer to datamodule.

If you want to override, here is an example
```
python research/train.py data=wsj/wsj_full dataloader@dataloader.default=constant_token_big -c job
```


If you set Trainer.accumulate_grad_batches, the size will be reduced automatically.
