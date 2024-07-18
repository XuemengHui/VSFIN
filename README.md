##### Quick Start Guide for Training
Modify the data path in  'data/build.py'

The data can be downloaded at :[CUB](https://drive.google.com/file/d/1bRbb6DzwWccwxhCkREuAadYL73jgVgfe/view?usp=share_link), [AwA2](https://drive.google.com/file/d/1ekXylwVbIY9QAbXmQe-Gwk1vk52qfEby/view?usp=share_link), [SUN](https://drive.google.com/file/d/1BEL_Sth2ZdmNaPBrF01Yub70xnIL6YlR/view?usp=share_link). 

Generate : Use ```extract_attribute_w2v_CUB.py``` ```extract_attribute_w2v_SUN.py``` ```extract_attribute_w2v_AWA2.py``` to generate the word embedding of semantic description and place it in ```datasets/Attribute/w2v```.

python -m torch.distributed.launch --nproc_per_node=2 train.py --config-file config/'dataset'.yaml

Experiments are performed with two RTX 3090Ti GPU.

## Model Architecture

Model Architecture will be uploaded when this paper is accepted.

## Requirements

Python 3.8.5
Torch  1.8.0+cu111

## Ackowledgement

This code package is based in part on the source code of the [GEM-ZSL] and [PSVMA]repository.

[GEM-ZSL] (https://github.com/osierboy/GEM-ZSL)
[PSVMA] (https://github.com/ManLiuCoder/PSVMA)
