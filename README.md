# qr_decoder

全 AI 编码, 不依赖任何高级库(除了图片读写, 以及常规的 opencv 操作)

单纯为学习 QR 解码怎么做. ai 的解法不一定和实际的一样, 但是鉴于大体可工作, 所以可认为这就是典型的做法.

怎么解码 qr_decode.py. 原理见 https://github.com/superzhangmch/qr_decoder/blob/main/qr_decode_explained.md 

----

网上随便找一个 QR code, 中间步骤如下: 找共心圈套圈, 作为角块. 三个角块, 可以由他们的相对位置推断出右下角(三个角块各自的四个顶点，共 12 个图像坐标, 用来推断右下角)

<img src='test/111_debug/1_detected.png' height=512/> <br>

定位出四个角后, 就可以把二维码还原成正方形了:

<img src='test/111_debug/2_warped.png' height=500 width=1500 /> <br> 

各区块的分布, 以及 payload 怎么读取的(右下角开始, z 字形 + 耕牛式前进):

<img src='test/111_debug/3_matrix.png' height=512 width=512/> <br>

取 XOR 后, 就可以读数据了(数据残缺要用 ReedSolomon 算法来纠正): <br>
<img src='test/111_debug/4_unmasked.png' height=512 width=512/> <br> 
 
对于中心的 logo, 设法把它排除, 用 ReedSolomon 就足以把 mask的补回来.
