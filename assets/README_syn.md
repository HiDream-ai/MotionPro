## MotionPro-Dense Video Generation Pipeline


This document provides an introduction to the MotionPro-Dense video generation pipeline, detailing its functionality and workflow. The pipeline is illustrated in the diagram below.

### Pipeline Description

1. **Video Generation with Base Motion Control**
   - First, MotionPro can be used to generate videos with controllable object motion and camera motion.
   
2. **Optical Flow and Visibility Mask Extraction and Merging**  
   - The generated videos are processed using CoTracker, a tool for extracting optical flow and visibility masks for each frame.  
   - The extracted optical flows are accumulated through summation.  
   - The per-frame visibility masks from both sequences are intersected to obtain the final visibility mask.

3. **Final Video Generation with Combined Motions**  
   - The aggregated motion conditions are used as input for **MotionPro-Dense**, which generates the final video with seamlessly integrated object and camera motions. 


<div align="center">
  <video src="./syn_video_control1.mp4" width="90%" autoplay loop muted playsinline poster="">
  </video>
  <p><em>Figure 1: Illustration of video generation with combined motions.</em></p>
</div>


### Synchronized Video Generation

Additionally, the pipeline enables the generation of **synchronized videos**, where a consistent object motion is paired with different camera motions. 

<div align="center">
  <video src="./syn_video_control2.mp4" width="90%" autoplay loop muted playsinline poster="">
  </video>
  <p><em>Figure 2: Illustration of synchronized video generation.</em></p>
</div>



