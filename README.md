# Anime style transfer model

## Goal

Given a face picture and two conditions (**sex**, **age**), the model will generate a anime-style face according to the face picture and the conditions.

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


## Reference

### [Paper](https://arxiv.org/abs/1907.10830)
