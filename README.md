# Upscalify (WiP)

This is a repository for my experiments with transformer-based image super-resolution methods for my research project at
the [Vytautas Magnus University](https://www.vdu.lt/en/).

## Methods

### SwinIR

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gQpcOjJKcIRAfQVG-DKGIe4NF53RiviA?usp=sharing)

Paper: [SwinIR: Image Restoration Using Swin Transformer](./papers/2108.10257v1-SwinIR.pdf)

#### Training

| Loss                                            | PSNR                                            | SSIM                                            |
|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| <img src="swinir/plots/loss.png" width="300px"> | <img src="swinir/plots/psnr.png" width="300px"> | <img src="swinir/plots/ssim.png" width="300px"> |

### SwinFIR

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n8fy4FICIEiuVPeFMN2ggpgC55H7N8_h?usp=sharing)

Paper: [SwinFIR: Revisiting the SwinIR with Fast Fourier Convolution and Improved Training for Image Super-Resolution](./papers/2208.11247v3-SwinFIR.pdf)

#### Training

| Loss                                             | PSNR                                             | SSIM                                             |
|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| <img src="swinfir/plots/loss.png" width="300px"> | <img src="swinfir/plots/psnr.png" width="300px"> | <img src="swinfir/plots/ssim.png" width="300px"> |

### HAT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GgzeO6rMeiI0nC1A6LQVTr_G2FRIS0Dp?usp=sharing)

Paper: [HAT: Hybrid Attention Transformer for Image Restoration](./papers/2309.05239v2-HAT.pdf)

#### Training

##### 2x Pre-training

| Loss                                            | PSNR                                            | SSIM                                            |
|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| <img src="hat/plots/loss_2x.png" width="300px"> | <img src="hat/plots/psnr_2x.png" width="300px"> | <img src="hat/plots/ssim_2x.png" width="300px"> |

<br/>

##### 4x Fine-tuning

| Loss                                            | PSNR                                            | SSIM                                            |
|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| <img src="hat/plots/loss_4x.png" width="300px"> | <img src="hat/plots/psnr_4x.png" width="300px"> | <img src="hat/plots/ssim_4x.png" width="300px"> |

### DRCT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16kOqEp7W7GFzL00-FJTc7LN8noRpphEe?usp=sharing)

Paper: [DRCT: Saving Image Super-Resolution away from Information Bottleneck](./papers/2404.00722v5-DRCT.pdf)

#### Training

##### 2x Pre-training

| Loss                                             | PSNR                                             | SSIM                                             |
|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| <img src="drct/plots/loss_2x.png" width="300px"> | <img src="drct/plots/psnr_2x.png" width="300px"> | <img src="drct/plots/ssim_2x.png" width="300px"> |

<br/>

##### 4x Fine-tuning

| Loss                                             | PSNR                                             | SSIM                                             |
|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| <img src="drct/plots/loss_4x.png" width="300px"> | <img src="drct/plots/psnr_4x.png" width="300px"> | <img src="drct/plots/ssim_4x.png" width="300px"> |

## Results

| **Method / Metric** | **PSNR (from paper)** | **SSIM (from paper)** | **PSNR (experiment)**                       | **SSIM (experiment)**                       |
|---------------------|-----------------------|-----------------------|---------------------------------------------|---------------------------------------------|
| **SwinIR**          | 32.72                 | 0.9021                | <span style="color:green">**32.986**</span> | <span style="color:green">**0.9023**</span> |
| **SwinFIR**         | 33.08                 | 0.9048                | <span style="color:red">**33.063**</span>   | <span style="color:red">**0.9036**</span>   |
| **HAT**             | 33.04                 | 0.9056                | <span style="color:green">**33.115**</span> | <span style="color:red">**0.9041**</span>   |
| **DRCT**            | 33.11                 | 0.9064                | <span style="color:red">**32.832**</span>   | <span style="color:red">**0.9008**</span>   |

**Table 1.** Results of experiments performing 4× super-resolution. *Note: The metrics were calculated using the Set5
dataset. Green values indicate results exceeding those published in the paper.*
