# Image_Retrieval_Challenge-
Sort images by correlations of the test image in Wiki and NUS-WIDE.

*   `config.py` 设置代码目录，并行线程数
*   `nus_dataset.py` 导入 nus 所有数据，同时从内存 dump 成文件储存
*   `wiki_dataset.py` 导入 wiki 所有数据，同时从内存 dump 成文件储存
*   `wiki_trainer.py` 导入 wiki 数据并且进行训练
*   `nus_trainer.py` 导入 wiki 数据并且进行训练
*   `nus_learner_lr.py` 使用 Logistic Regression
*   `nus_learner_lr_bs.py` 使用 Logistic Regression + booststrap
