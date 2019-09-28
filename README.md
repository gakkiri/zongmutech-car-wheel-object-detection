# zongmutech-car-wheel-object-detection

# 目录结构
---data  
  |   |-train  
  |   |-train_label  
  |   |-test  
  |   |-test_label  
  |  
  |-ens-...  
  |-data_test  

# 用法
1. 需求
```
pip install -r requirements.txt
```
2. 按目录结构准备数据
3. 将训练模型放在./checkpoints中
4. 运行代码得到submit.txt
```
python inference_ens.py --image_folder='../data_test'
```
5. 运行代码将txt文件转化成xml文件, 得到的结果存放在inference_xml中
```
python submit_xml.py
```
# 推理结果
![result]https://github.com/joinssmith/zongmutech-car-wheel-object-detection/blob/master/imgs/result.png

# 模型
![model]https://github.com/joinssmith/zongmutech-car-wheel-object-detection/blob/master/imgs/model.png
