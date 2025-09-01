<div align="center">
<img src="logo.png" alt="DEQ-RWKV的LOGO" />

  # DEQ-RWKV
  
  <p>结合深度均衡模型(DEQ)与RWKV-v7架构的轻量级AI模型</p>
  
  <div>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
    <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
    <img src="https://img.shields.io/badge/OpenSource-%E2%9D%A4%EF%B8%8F-green?style=for-the-badge" />
  </div>
</div>

这是一个实验和学习性质的开源项目，把**DEQ**（深度均衡模型）算法和**RWKV-v7**架构结合起来玩一玩🤗

---

## 项目是做什么的 🤔

简单来说，这个项目尝试用更少的模型参数来做和RWKV-v7差不多的事，我用DEQ算法来优化模型结构，看看能不能在保持性能的同时，让模型更轻量一些。

<div class="highlight-box">
  <h3>💡 多轻量？</h3>
  <ul>
    <li>传统RWKV-v7：1个768维嵌入的block ≈ <strong>28.7M</strong> 文件大小</li>
    <li>官方最小的0.1B模型：12个block ≈ <strong>344M</strong> 文件大小</li>
    <li><strong>DEQ-RWKV</strong>：仅需 <strong>1个block</strong> ，你想想🙃</li>
  </ul>
</div>

**显存优化**：由于DEQ特性，中间的多个block只需存一个block的梯度，大幅降低训练的显存占用

---

## 主要文件有什么用 📁

| 文件/目录        | 功能描述                                            |
|-----------------|--------------------------------------------------|
| **main.ipynb**  | 完整的模型定义和训练流程Jupyter笔记本               |
| **wkv7.py**     | CUDA实现的RWKV-v7的wkv模块Python封装接口             |
| **tokenizer.py**| RWKV官方的分词器，负责文本与token的相互转换           |
| **vocab.txt**   | RWKV的官方词表                          |
| **cuda/**       | 里面是用CUDA写的核心计算代码，也是RWKV官方的         |
| **test.jsonl**  | 一个测试训练效果用的数据集，里面是一些已经转成token的数据 |

注意：main.py里的用到的g1a.pth是rwkv7的官方的某个0.1b模型的参数，可自行下载（哪个都可以），放在主目录就好

<a href="https://huggingface.co/BlinkDL/rwkv7-g1/tree/main" target="_blank">
  <img src="https://img.shields.io/badge/Hugging%20Face-%F0%9F%A4%97-blue?style=for-the-badge" />
</a>

---

## 怎么用 🚀

如果你想自己捣鼓这个项目，或者学习DEQ算法如何用，直接看代码就好啦～ 训练流程在main.ipynb里写得很清楚（只不过我没怎么写注释😅）

---

## 为什么开源 🌟

就是想分享一下我实验了很久的代码😅  
如果你觉得有用，可以自己拿去改，或者在这个基础上做更多的实验。   
有什么问题或者建议，也欢迎提出来一起讨论   

---

## 致谢 🙏

- 🖥️感谢 **[Mingkai Zhou](https://github.com/sg-first)** 赞助的算力支持！
- 🙌感谢 RWKV 社区提供的开源代码！
