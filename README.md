# Anime style transfer model

## Goal

Given a face picture and two conditions (**age**, **sex**), the model will generate a anime-style face according to the face picture and the conditions.

![](/utils/result_1.png)

## Netwotk architechture

![](/utils/network.png)

## Usage

### Train
```
> python main.py --dataset selfie2anime
```
* If the memory of gpu is **not sufficient**, set `--light` to True

### Test
```
> python main.py --dataset selfie2anime --phase test
```

## Results

![](/utils/result_all.png)

- [Report](/utils/report.pdf)

## Reference

- [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)
- [U-GAT-IT Official PyTorch Implementation](https://github.com/znxlwm/UGATIT-pytorch)
- [Anime Planet](https://www.anime-planet.com/)
- dataset (human face with age & gender)
  - [Source 1](https://github.com/JingchunCheng/All-Age-Faces-Dataset)
  - [Source 2](https://susanqq.github.io/UTKFace/)

## Partner

羅子涵 Github@hedy881028
