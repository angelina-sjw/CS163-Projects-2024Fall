---
layout: post
comments: true
title: Vehicle Trajectory Prediction
author: Angelina Sun, Jinyuan Zhang, Jun Yu Chen
date: 2024-12-13
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## CRAT-Pred

Earlier, the solutions we explore utilize spatial and temporal dynamics but rely heavily on rasterized map data. CRAT-Pred introduces a novel approach to trajectory prediction, leveraging Crystal Graph Convolutional Neural Networks (CGCNNs) and Multi-Head Self-Attention (MHSA) in a map-free, interaction-aware framework.

CRAT-Pred achieves high performance by combining temporal and spatial modeling with efficient graph-based and attention mechanisms. Let’s explore the architecture and methodology in detail.

### Architectural Details

![CRAT-Pred Architecture Overview]({{ '/assets/images/team26/crat_architecture.png' | relative_url }})


CRAT-Pred begins by encoding temporal information of vehicle trajectories. Each vehicle’s motion history is represented as a sequence of displacements and visibility flags. The displacement vector $$\Delta \tau_i^t = \tau_i^t - \tau_i^{t-1}$$ captures the positional change between successive time steps, and a binary flag $$b_i^t$$ indicates whether the vehicle is visible at time $$ t$$. Together, they form the input representation $$s_i^t = (\Delta \tau_i^t \| b_i^t)$$. These inputs are processed by a shared LSTM layer that encodes temporal dependencies for all vehicles. The hidden state of the LSTM, $$h_i^t = \text{LSTM}(h_i^{t-1}, s_i^t; W_{\text{enc}}, b_{\text{enc}})$$, summarizes the motion history into a compact, 128-dimensional representation.

The next stage involves modeling spatial interactions between vehicles using a fully connected graph. Each vehicle serves as a node, initialized with its LSTM-encoded hidden state, $$v_i^{(0)} = h_i^0$$. Edges in the graph capture pairwise distances between vehicles, defined as $$e_{i,j} = \tau_j^0 - \tau_i^0$$. The Crystal Graph Convolutional Network (CGCNN) updates these node features while incorporating edge information. The graph convolution operation is given by $$v_i^{(g+1)} = v_i^{(g)} + \sum_{j \in N(i)} \sigma(z_{i,j}^{(g)} W_f^{(g)} + b_f^{(g)}) \cdot g(z_{i,j}^{(g)} W_s^{(g)} + b_s^{(g)})$$, where $$z_{i,j}^{(g)} = [v_i^{(g)} \| v_j^{(g)} \| e_{i,j}]$$ is the concatenation of node and edge features. This design allows the model to capture spatial dependencies effectively. Two graph convolution layers are used, each followed by batch normalization and ReLU activations.

To further refine the understanding of interactions, CRAT-Pred applies a multi-head self-attention mechanism. This mechanism computes pairwise attention scores that indicate the influence of one vehicle on another. For each attention head, the output is calculated as $$\text{head}_h = \text{softmax}\left(\frac{V^{(g)} Q_h (V^{(g)} K_h)^\top}{\sqrt{d}}\right) V^{(g)} V_h$$, where $$Q_h, K_h,$$ and $$V_h$$ are linear projections of the node features, and $$d$$ is a scaling factor based on the embedding size. Outputs from all heads are concatenated and transformed as $$A = \left[\text{head}_1 \| \dots \| \text{head}_{L_h}\right] W_o + b_o$$, resulting in a 128-dimensional interaction-aware embedding for each vehicle. The attention weights provide interpretable measures of interaction strength between vehicles.

The trajectory prediction is performed by a decoder that uses residual connections to refine the output. The decoder predicts positional offsets relative to the initial position of the vehicle, defined as $$o_i = \left(\text{ReLU}\left(F(a_i; \{W_r, b_r\}) + a_i\right)\right) W_{\text{dec}} + b_{\text{dec}}$$. Here, $$F(a_i)$$ applies non-linear transformations to the interaction-aware features $$a_i$$, while the residual connection enhances stability and performance. CRAT-Pred supports multi-modality by using multiple decoders to predict diverse plausible trajectories. These decoders are trained using Winner-Takes-All (WTA) loss, which optimizes only the decoder with the lowest error for each sequence, ensuring diversity in the predictions without complicating the training process.

### Training and Evaluation

The training of CRAT-Pred follows a two-stage process. Initially, the model is trained end-to-end with a single decoder using smooth-L1 loss to predict the most probable trajectory. Once this phase converges, additional decoders are introduced, and the model is fine-tuned using WTA loss to handle multi-modal predictions. This approach ensures that CRAT-Pred can produce diverse yet accurate predictions.

CRAT-Pred is evaluated on the Argoverse Motion Forecasting Dataset, which provides large-scale, real-world vehicle trajectory data. The performance metrics include minimum Average Displacement Error (minADE), minimum Final Displacement Error (minFDE), and Miss Rate (MR). These metrics measure the average error of predicted trajectories, the final displacement error, and the percentage of predictions that miss the ground truth by more than 2 meters, respectively. CRAT-Pred achieves state-of-the-art results among map-free models, outperforming many competitors even with significantly fewer model parameters.

![CRAT-Pred Argoverse results]({{ '/assets/images/team26/crat_results.png' | relative_url }})

The qualitative results of CRAT-Pred on the Argoverse validation set are presented above for three diverse sequences. The past observed trajectory of the target vehicle is depicted in blue, while the ground-truth future trajectory is shown in green. Predicted trajectories are illustrated in orange and red, with orange indicating the most probable future trajectory. The past trajectories of other vehicles are visualized in purple. For context, road topologies, though not utilized by the prediction model, are displayed using dashed lines.

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
