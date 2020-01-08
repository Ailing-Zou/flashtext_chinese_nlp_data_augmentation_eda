# flashtext-chinese
这个repository的目的是进行NLP领域的数据增强，参考的论文是EDA参考文献：EDA Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks flashtext for chinese ，2019.8
数据增强包括四种方法，同义词替换、随机插入、随机交换、随机删除，而同义词的中文字典来自于https://github.com/Keson96/SynoCN.
在进行数据增强模式时，由于本身的数据和同义词的数据量都比较大，因此需要进行高效的匹配，而非正则表达式的匹配，进而采用了flashtext方法。





