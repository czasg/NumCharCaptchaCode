## 数字字符型验证码识别

目前主要采用卷积神经网络模型

#### 项目结构
* `/model`: 用于存放模型，目前主要提供cnn卷积神经网络模型
* `/trainModel`: 用于存放训练模型，每一个子目录是一个小项目，目前提供搜狗微信、知道创宇等字符验证码的训练代码
    * `/ModelManager.py`: 用于收集已训练好的模型，并提供对一致的接口，需要在`ModelManager`中先行注册。接口的实现可以看源码
* `/app.py`: 基于tornado搭建的后端服务，对接 `/trainModel/ModelManager.py` 接口服务

##### 如何执行
以 `/trainModel/demo` 来说


