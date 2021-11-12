把numpy array变成torch tensor，然后把rgb值归一到[0, 1]这个区间

root是存储数据的文件夹，download=True指定如果数据不存在先下载数据

在__init__中定义网络需要的操作算子

Conv2d的第一个参数是输入的channel数量，第二个是输出的channel数量，第三个是kernel size

由于上一层有16个channel输出，每个feature map大小为5*5，所以全连接层的输入是16*5*5

最终有10类，所以最后一个全连接层输出数量是10

使用torch.no_grad的话在前向传播中不记录梯度，节省内存

随着Batch_Size增大，处理相同数据量的速度越快。 
随着Batch_Size 增大，达到相同精度所需要的epoch 数量越来越多。 
由于上述两种因素的矛盾， Batch_Size 增大到某个时候，达到时间上的最优。 
由于最终收敛精度会陷入不同的局部极值，因此Batch_Size增大到某些时候，达到最终收敛精度上的最优。
