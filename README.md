# Deep-Learning 深度學習-自然語言處理-新聞文件分類Document-Classification

深度學習-自然語言處理-新聞文件分類

這篇文章為 使用Tensorflow的Keras進行文件分類

需要先用pip install or conda install 安裝環境套件

我的tensorflow是2.2.0版本
python是3.8 

然後下載dataset放到正確的路徑即可

這邊放上dataset的下載連結 https://drive.google.com/file/d/1bLVmAwKd_GtMLsU9-rdLPvtC3Qiwya_b/view?usp=sharing

還有pre-trained model的下載連結 https://drive.google.com/drive/folders/1QEbdEn-DO-23hYCLgaB3f4vP8e8L8iIp?usp=sharing

環境安裝好後即可python document_classification.py執行程式

模型訓練流程:
1. Dataset: 網路下載

2. 特徵萃取: Word2Vec

3. 特徵選擇: 卷積神經網路(CNN)

4. 分類氣: 3-Layers Fully Connect

------------------------------------------------------------------------
# 延伸閱讀 Read-Around
可以試一試使用不同的語言模型進行訓練

不同的語言模型將會有不一樣的成效

各式語言模型:https://github.com/Tsai-Cheng-Hong/Deep-Learning-Word-Embedding
