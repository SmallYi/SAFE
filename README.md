# SAFE : Self Attentive Function Embedding

Paper
---
This software is the outcome of our accademic research. See our arXiv paper: [arxiv](https://arxiv.org/abs/1811.05296)

If you use this code, please cite our accademic paper as:

```bibtex
@inproceedings{massarelli2018safe,
  title={SAFE: Self-Attentive Function Embeddings for Binary Similarity},
  author={Massarelli, Luca and Di Luna, Giuseppe Antonio and Petroni, Fabio and Querzoni, Leonardo and Baldoni, Roberto},
  booktitle={Proceedings of 16th Conference on Detection of Intrusions and Malware & Vulnerability Assessment (DIMVA)},
  year={2019}
}
```

What you need  
-----
You need [radare2](https://github.com/radare/radare2) installed in your system. [Install reference](https://blog.csdn.net/weixin_40732417/article/details/105586107)</br>
 [win_radare2_install](https://bbs.pediy.com/thread-225529.htm)</br>
 [官网](https://rada.re/r/)
 ```python
git clone https://github.com/radareorg/radare2
cd radare2 ; sys/install.sh
 ```
  
Quickstart
-----
To create the embedding of a function:
```
git clone https://github.com/gadiluna/SAFE.git
pip install -r requirements
chmod +x download_model.sh
./download_model.sh #需要访问外网
python safe.py -m data/safe.pb -i helloworld.o -a 100000F30 #-i是目标二进制，-a是目标函数地址
```
#### What to do with an embedding?
Once you have two embeddings ```embedding_x``` and ```embedding_y``` you can compute the similarity of the corresponding functions as: 
```
from sklearn.metrics.pairwise import cosine_similarity

sim=cosine_similarity(embedding_x, embedding_y)
 
```


Data Needed
-----
SAFE needs few information to work. Two are essentials, a model that tells safe how to 
convert assembly instructions in vectors (i2v model) and a model that tells safe how
to convert an binary function into a vector.
Both models can be downloaded by using the command
```
./download_model.sh
```
the downloader downloads the model and place them in the directory data.
The directory tree after the download should be.
```
safe/-- githubcode
     \
      \--data/-----safe.pb
               \
                \---i2v/
            
```
The safe.pb file contains the safe-model used to convert binary function to vectors.
The i2v folder contains the i2v model. 


Hardcore Details
----
This section contains details that are needed to replicate our experiments, if you are an user of safe you can skip
it. 

### Safe.pb
This is the freezed tensorflow trained model for AMD64 architecture. You can import it in your project using:

```
 import tensorflow as tf
 
 with tf.gfile.GFile("safe.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

 with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)
    
 sess = tf.Session(graph=graph)
``` 

see file: neural_network/SAFEEmbedder.py

### i2v
The i2v folder contains two files. 
A Matrix where each row is the embedding of an asm instruction.
A json file that contains a dictonary mapping asm instructions into row numbers of the matrix above.
see file: asm_embedding/InstructionsConverter.py



## Train the model
If you want to train the model using our datasets you have to first use:
```
 python downloader.py -td # 需要访问外网
```
This will download the datasets into data folder. Note that the datasets are compressed so you have to decompress them yourself.
This data will be an sqlite databases.
To start the train use neural_network/train.sh.
The db can be selected by changing the parameter into train.sh.
If you want information on the dataset see our paper.
```python
# 使用作者的数据训练模型
cd neural_network 
./train.sh
```

## Create your own dataset
If you want to create your own dataset you can use the script ExperimentUtil into the folder
dataset creation.
```python
(venv-js)python ./dataset_creation/ExperimentUtil.py # 创建自己的数据集
```

## Create a functions knowledge base
If you want to use SAFE binary code search engine you can use the script ExperimentUtil to create
the knowledge base.
Then you can search through it using the script into function_search
```python
(venv-js)python ./dataset_creation/ExperimentUtil.py # 创建函数知识库
function_search # 函数搜索
```


Related Projects
---

* YARASAFE: Automatic Binary Function Similarity Checks with Yara (https://github.com/lucamassarelli/yarasafe) 
* SAFEtorch: Pytorch implemenation of the SAFE neural network (https://github.com/facebookresearch/SAFEtorch)

Thanks
---
In our code we use [godown](https://github.com/circulosmeos/gdown.pl) to download data from Google drive. We thank 
circulosmeos, the creator of godown.

We thank Davide Italiano for the useful discussions. 

Run_Record
---
1. python2 import tensorflow很慢
2. 作者的代码使用python3版本
3. tensorflow2.0提示错误：module 'tensorflow' has no attribute 'placeholder'<br/>
import tensorflow as tf 替换为<br/>
import tensorflow.compat.v1 as tf<br/>
tf.disable_v2_behavior()<br/>
4. 使用tensorflow==1.14.0
5. tensorflow屏蔽版本更迭的warning<br/>
tf.logging.set_verbosity(tf.logging.ERROR)<br/>
6. 输出日志目录/mnt/jiangs/SAFE_Pro/SAFE_Core/data/experiments/openssl/out/runs保存了模型的大文件，比较耗时
7. Tqdm 是一个快速，可扩展的Python进度条
8. 在window安装好radare2后要配置环境变量，并重启pycharm才能生效
9. 原生程序需要的训练数据是.o类型的
10. 原生程序适用于linux环境，如os.setpgrp()
11. 直接在linux系统中用git下载代码，在windows下载代码再上传到linux系统会导致文件内容编码有错误
