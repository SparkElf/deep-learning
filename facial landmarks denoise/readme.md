1. mediapipe-python采用protobuf协议，和普通的python库不同，数据结构的转换上有一些麻烦，只能通过new class后赋值的方式完成python数据结构到mediapipe数据结构的转换。
2. cv2.videowriter只支持np.uint8格式，并且必须在程序结束前执行release函数
3. self2self方法存在边际噪音，对于图像，图像的边框附近噪声强烈，对于时序数据，开始和结束的时刻噪声强烈
4. mediapipe x、y是归一化到[0,1]范围的数据，z不是，分布不同，所以需要分开处理。而xy分布相同且具有相关性，可以合在一起滤波处理
5. self2self的泛化能力较弱，基本不具有泛化能力，应该往浅层、小网络发展
6. dropout一般不在浅层网络使用
7. 浅层网络不一定学习率就小