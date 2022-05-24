
# Nested NER as Latent Lexicalized Parsing

The code of our ACL2022 paper: [Nested Named Entity Recognition as Latent Lexicalized Constituency Parsing](https://aclanthology.org/2022.acl-long.428/). 

## Prerequisits

The code was implemented in Python 3.9 with anaconda environment. See `requirements.txt` file for all dependencies. The default experiment configurations assume at least 24GB GPU memory.

## Data

Prepare data using [Shibuya's repo](https://github.com/yahshibu/nested-ner-tacl2020). The processed data looks like this
```
WOODRUFF We know that some of the American troops now fighting in Iraq are longtime veterans of warfare , probably not most , but some .
1,2 ORG|4,13 PER|6,13 PER|7,8 GPE|12,13 GPE|14,18 PER|21,22 PER|24,25 PER

Their military service goes back to the Vietnam era .
0,1 PER|1,2 ORG|7,8 GPE

...
```

Next we will convert the data to the brat standoff format. See [its document](https://brat.nlplab.org/standoff.html) for more information. We provide a script for the conversion in the `data` folder.
Run it with command
```shell
python raw2brat.py -o <output_folder> <input_file>
```

## Train

```
python train.py model=<dataset_name>
```
`<dataset_name>` is one of `ace05`, `ace04`, `genia` and `nne`.

## Test

```
python test.py runner.load_from_checkpoint=<path_to_ckpt>
```
It will report scores and write the predictions on the test set to the `predict_on_test` file.

## Citation

```
@inproceedings{lou-etal-2022-nested,
    title = "Nested Named Entity Recognition as Latent Lexicalized Constituency Parsing",
    author = "Lou, Chao  and
      Yang, Songlin  and
      Tu, Kewei",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.428",
    pages = "6183--6198",
    abstract = "Nested named entity recognition (NER) has been receiving increasing attention. Recently, Fu et al. (2020) adapt a span-based constituency parser to tackle nested NER. They treat nested entities as partially-observed constituency trees and propose the masked inside algorithm for partial marginalization. However, their method cannot leverage entity heads, which have been shown useful in entity mention detection and entity typing. In this work, we resort to more expressive structures, lexicalized constituency trees in which constituents are annotated by headwords, to model nested entities. We leverage the Eisner-Satta algorithm to perform partial marginalization and inference efficiently.In addition, we propose to use (1) a two-stage strategy (2) a head regularization loss and (3) a head-aware labeling loss in order to enhance the performance. We make a thorough ablation study to investigate the functionality of each component. Experimentally, our method achieves the state-of-the-art performance on ACE2004, ACE2005 and NNE, and competitive performance on GENIA, and meanwhile has a fast inference speed.",
}
```