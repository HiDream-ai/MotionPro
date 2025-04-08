# MotionPro

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    🖥️ <a href="https://github.com/HiDream-ai/MotionPro">GitHub</a> &nbsp&nbsp ｜ &nbsp&nbsp  🌐 <a href="https://zhw-zhang.github.io/MotionPro-page/"><b>Project Page</b></a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/HiDream-ai/MotionPro/tree/main">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="">Paper </a> &nbsp&nbsp | &nbsp&nbsp 📖 <a href="">PDF</a> &nbsp&nbsp 
<br>
## Please wait for paper release....
[**MotionPro: A Precise Motion Controller for Image-to-Video Generation**](https://zhw-zhang.github.io/MotionPro-page/) <be>

🔆 If you find MotionPro useful, please give a ⭐ for this repo, which is important to Open-Source projects. Thanks!

In this repository, we introduce **MotionPro**, an image-to-video generation model built on SVD. MotionPro learns object and camera motion control from **in-the-wild** video datasets (e.g., WebVid-10M) without applying special data filtering. The model offers the following key features:

-  **User-friendly interaction.** Our model requires only simple conditional inputs, allowing users to achieve I2V motion control generation through brushing and dragging.
-  **Simultaneous control of object and camera motion.** Our trained MotionPro model supports simultaneous object and camera motion control. Moreover, our model can achieve precise camera control driven by pose without requiring training on a specific camera-pose paired dataset. [More Details](assets/camera_control.png)
-  **Synchronized video generation.** This is an extension of our model. By combining MotionPro and MotionPro-Dense, we can achieve synchronized video generation. [More Details](assets/README_syn.md)


Additionally, our repository provides more tools to benefit the research community's development.:

-  **Memory optimization for training.** We provide a training framework based on PyTorch Lightning, optimized for memory efficiency, enabling SVD fine-tuning with a batch size of 8 per NVIDIA A100 GPU.
-  **Data construction tools.**  We offer scripts for constructing training data. Additionally, we also provide code for loading datasets in two formats, supporting video input from both folders (Dataset) and tar files (WebDataset).
-  **MC-Bench and evaluation code.** We constructed MC-Bench with 1.1K user-annotated image-trajectory pairs, along with evaluation scripts for comprehensive assessments. All the images showcased on the project page can be found here.

## Video Demos

<div align="center">
  <video controls autoplay loop muted playsinline src="https://github.com/user-attachments/assets/2af6d638-e09c-4e98-a565-43c8ca30f91b"></video>
  <p><em>Examples of different motion control types by our MotionPro.</em></p>
</div>

## 🔥 Updates
- [x] **\[2025.03.26\]** Release inference and training code.
- [ ] **\[2025.04.08\]** Release MC-Bench and evaluation code.
- [ ] Upload gradio demo usage video.
- [ ] Upload annotation tool for image-trajectory pair construction.

## 🏃🏼 Inference
<details open>
<summary><strong>Environment Requirement</strong></summary>

Clone the repo:
```
git clone https://github.com/HiDream-ai/MotionPro.git
```

Install dependencies:
```
conda create -n motionpro python=3.10.0
conda activate motionpro
pip install -r requirements.txt
```
</details>

<details open>
<summary><strong>Model Download</strong></summary>


| Models            | Download Link                                                                 | Notes                                      |
|-------------------|-------------------------------------------------------------------------------|--------------------------------------------|
| MotionPro  | 🤗[Huggingface](https://huggingface.co/HiDream-ai/MotionPro/blob/main/MotionPro-gs_16k.pt)                | Supports both object and camera control. This is the default model mentioned in the paper.   |
| MotionPro-Dense   | 🤗[Huggingface](https://huggingface.co/HiDream-ai/MotionPro/blob/main/MotionPro_Dense-gs_14k.pt)           | Supports synchronized video generation when combined with MotionPro. MotionPro-Dense shares the same architecture as Motion, but the input conditions are modified to include: dense optical flow and per-frame visibility masks relative to the first frame. |


Download the model from HuggingFace at high speeds (30-75MB/s):
```
cd tools/huggingface_down
bash download_hfd.sh
```
</details>


<details open>
<summary><strong>Run Motion Control</strong></summary>

This section of the code supports simultaneous object motion and camera motion control. We provide a user-friendly Gradio demo interface that allows users to control motion with simple brushing and dragging operations. The instructional video can be found in `assets/demo.mp4` (please note the version of gradio).

```
python demo_sparse_flex_wh.py
```
When you expect all pixels to move (e.g., for camera control), you need to use the brush to fully cover the entire area. You can also test the demo using `assets/logo.png`.

Additionally, users can also generate controllable image-to-video results using pre-defined camera trajectories. Note that our model has not been trained on a specific camera control dataset. Test the demo using `assets/sea.png`.

```
python demo_sparse_flex_wh_pure_camera.py
```
</details>


<details open>
<summary><strong>Run synchronized video generation and video recapture</strong></summary>

By combining MotionPro and MotionPro-Dense, we can achieve the following functionalities:
- Synchronized video generation. We assume that two videos, `pure_obj_motion.mp4` and `pure_camera_motion.mp4`, have been generated using the respective demos. By combining their motion flows and using the result as a condition for MotionPro-Dense, we obtain `final_video`. By pairing the same object motion with different camera motions, we can generate `synchronized videos` where the object motion remains consistent while the camera motion varies. [More Details](assets/README_syn.md)

Here, you need to first download the [model_weights](https://huggingface.co/HiDream-ai/MotionPro/blob/main/tools/co-tracker/checkpoints/scaled_offline.pth) of cotracker and place them in the `tools/co-tracker/checkpoints` directory.

```
python inference_dense.py --ori_video 'assets/cases/dog_pure_obj_motion.mp4' --camera_video 'assets/cases/dog_pure_camera_motion_1.mp4' --save_name 'syn_video.mp4' --ckpt_path 'MotionPro-Dense CKPT-PATH'
```

</details>

## 🚀 Training

<details open>
<summary><strong>Data Prepare</strong></summary>

We have packaged several demo videos to help users debug the training code. Simply 🤗[download](https://huggingface.co/HiDream-ai/MotionPro/tree/main/data), extract the files, and place them in the `./data` directory.

Additionally, `./data/dot_single_video` contains code for processing raw videos using [DOT](https://github.com/16lemoing/dot) to generate the necessary conditions for training, making it easier for the community to create training datasets.

</details>


<details open>
<summary><strong>Train</strong></summary>

Simply run the following command to train MotionPro:
```
bash train_server_1.sh
```
In addition to loading video data from folders, we also support [WebDataset](https://rom1504.github.io/webdataset/), allowing videos to be read directly from tar files for training. This can be enabled by modifying the config file:
```
train_debug_from_folder.yaml -> train_debug_from_tar.yaml 
```

Furthermore, to train the **MotionPro-Dense** model, simply modify the `train_debug_from_tar.yaml` file by changing `VidTar` to `VidTar_all_flow` and updating the `ckpt_path`.


## 📝Evaluation


<summary><strong>MC-Bench</strong></summary>

Simply download 🤗[MC-Bench](https://huggingface.co/HiDream-ai/MotionPro/blob/main/data/MC-Bench.tar), extract the files, and place them in the `./data` directory.

<summary><strong>Run eval script</strong></summary>

Simply execute the following command to evaluate MotionPro on MC-Bench and Webvid:
```
bash eval_model.sh
```
</details>

## 🌟 Star and Citation
If you find our work helpful for your research, please consider giving a star⭐ on this repository and citing our work.
```
@inproceedings{2025motionpro,
 title={{MotionPro: A Precise Motion Controller for Image-to-Video Generation}},
 author={Zhongwei Zhang and Fuchen Long and Zhaofan Qiu and Yingwei Pan and Wu Liu and Ting Yao and Tao Mei},
 booktitle={CVPR},
 year={2025}
}
```


## 💖 Acknowledgement
<span id="acknowledgement"></span>

Our code is inspired by several works, including [SVD](https://github.com/Stability-AI/generative-models), [DragNUWA](https://github.com/ProjectNUWA/DragNUWA), [DOT](https://github.com/16lemoing/dot), [Cotracker](https://github.com/facebookresearch/co-tracker). Thanks to all the contributors! 

