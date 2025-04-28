Prediction of drug-disease interactions based on F-test and KSU resampling techniques
=
Enviroment
--
python: 3.1
matplotlib==3.6.2
numpy==1.26.4
pandas==2.1.1
scikit-learn==1.3.2
lime==0.2.0.1
PyQt5==5.15.9

Instruction
=

The function of the directory and the files in the directory are described as follows:

代码:

This directory holds the core code for this study.

数据及模型: 
This directory holds the datasets used in this study.

药物-疾病相互作用课题:

In this directory contains drug-disease data.

About Predictor
=
Download and Setup
--
The predictor can be downloaded at  https://pan.baidu.com/s/15Ifynpi2r_ABVGUNdaYMug?pwd=tg9x 提取码: tg9x
Download and get a zip package，unzip it ，find the predictor.exeand click it to run the predictor.
To facilitate online prediction, we have developed an online predictor based on Python.

How to Use Predictor
--
First, click the "Select Model" button at the top of the interface and choose the local model file. The file format should be "pkl".
![图片](https://github.com/user-attachments/assets/2194cc80-ec8f-445c-9cb5-0c4b1635cf0b)
After selecting the model file, choose the prediction file. Click the "Select Data" button at the top of the interface and choose the local data file. The file format should be "csv".
![图片](https://github.com/user-attachments/assets/85a7bb53-d827-4a25-8cde-f69dd6a4509f)
After selecting the test file, click "Start Prediction" and wait for the prediction results. The results will be displayed in a list format on the interface, as shown in Figure 5-6. The list contains 4 columns: the first column is the drug ID, the second is the disease ID, the third column is the predicted score, and the fourth column is the predicted label, represented as "negative" or "positive".
![图片](https://github.com/user-attachments/assets/c27c8203-967d-47ca-bd2d-9457416df253)
There is a button labeled "Save Results" at the top of the interface. By clicking this button, users can easily save the model's prediction results in CSV file format to their local computer.
![图片](https://github.com/user-attachments/assets/a933c917-f798-4961-b156-633dff057a88)




