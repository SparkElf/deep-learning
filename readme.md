1. Large pictures must be cropped. Because first of all, it may cause the overflow of memory and not be able to input pictures in batches; Secondly, the image is too large, which makes it difficult for visualization tools to present the original image result in the difficulty to observe the noise; Finally, the cropped picture can expand the training data, and the small size input makes the study easier.
2. Some experiences obtained from Unet: the data of the two feature layers can be fused by splicing, adding, multiplying, convolution and other ways. For image segmentation, the splicing performance is better.
3. Try to only use crop,avoid scale
4. The convolution layer has $ in\_channels*out\_channels*kernel\_size^2+out\_channels $ weight parameters and a bias parameter.[UNet parameters calculation](https://blog.csdn.net/jxb727098/article/details/118914245)
5. Replace linear layer by upsample and convolution net is recommended.
6. Don't forget to clear the gradient left by the previous data before each training session using optimizer.zero_grad()
7. Network could not learn info using large kernel like 8x8、16x16、32x32、64x64 etc,but work well using 3x3 kernel and deep pipline.
8. transform.ToTensor()是将输入的数据shape H，W，C ——> C，H，W;将所有数除以255，将数据归一化到【0，1】
   Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 将数据归一化到-1,1;Normalize(channel_mean, channel_std)，才是均值为0，标准差为1
9. Image.open(img_path).convert('RGB')比torchvision.io.read_image()好
10. 先预训练16个网络处理64x64图片的内容，再训练1个网络处理16个网络处理后的256x256的内容，反馈微调前面16个网络的内容
11. 一次实验意外：输入train(dncnn1(x),target)，train(dncnn2(model(x)),target),train(dncnn3(model(x)),target)
   y=dncnn1->dncnn2->dncnn3，效果还不错
12. 编码器解码器架构搭配norm有纯色斑点、光斑，原因是网络会创造一个信号峰来绕过norm，去掉任何norm psnr很高，但是细节观感不好
13. 输出层不需要任何激活函数，效果更好
14. 编解码器残差网络的优越性在于递进性：浅层网络解决不了的问题，剩下的残差应该更复杂，应该用更深的网络解决