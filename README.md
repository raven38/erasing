# Erasing Concepts from Diffusion Models
###  [Project Website](https://erasing.baulab.info) | [Arxiv Preprint](https://arxiv.org/pdf/2303.07345.pdf) | [Fine-tuned Weights](https://erasing.baulab.info/weights/esd_models/) | [Demo](https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion) <br>

### Updated code 🚀 - Now support diffusers!! (Faster and cleaner)

<div align='center'>
<img src = 'images/applications.png'>
</div>

Motivated by recent advancements in text-to-image diffusion, we study erasure of specific concepts from the model's weights. While Stable Diffusion has shown promise in producing explicit or realistic artwork, it has raised concerns regarding its potential for misuse. We propose a fine-tuning method that can erase a visual concept from a pre-trained diffusion model, given only the name of the style and using negative guidance as a teacher. We benchmark our method against previous approaches that remove sexually explicit content and demonstrate its effectiveness, performing on par with Safe Latent Diffusion and censored training.

To evaluate artistic style removal, we conduct experiments erasing five modern artists from the network and conduct a user study to assess the human perception of the removed styles. Unlike previous methods, our approach can remove concepts from a diffusion model permanently rather than modifying the output at the inference time, so it cannot be circumvented even if a user has access to model weights

Given only a short text description of an undesired visual concept and no additional data, our method fine-tunes model weights to erase the targeted concept. Our method can avoid NSFW content, stop imitation of a specific artist's style, or even erase a whole object class from model output, while preserving the model's behavior and capabilities on other topics.

## Code Update 🚀
We are releasing a cleaner code for ESD with diffusers support. Compared to our old-code this version uses almost half the GPU memory and is 5-8 times faster. Because of this diffusers support - we believe it allows to generalise to latest models (FLUX ESD coming soon ... ) <br>

To use the older version please go to `oldcode_erasing_compvis/` folder in this repository [here](https://github.com/rohitgandikota/erasing/tree/main/oldcode_erasing_compvis)

## Installation Guide
We recently updated our codebase to be much more cleaner and faster. The setup is also simple
```
git clone https://github.com/rohitgandikota/erasing.git
cd erasing
pip install -r requirements.txt
```

## Training Guide

After installation, follow these instructions to train a custom ESD model. Pick from following `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'`:
```
python esd_diffusers.py --erase_concept 'Van Gogh' --train_method 'xattn'
```

You can now erase an attribute from a concept!! 
```
python esd_diffusers.py --erase_concept 'red' --erase_from 'rose' --train_method 'xattn'
```

The optimization process for erasing undesired visual concepts from pre-trained diffusion model weights involves using a short text description of the concept as guidance. The ESD model is fine-tuned with the conditioned and unconditioned scores obtained from frozen SD model to guide the output away from the concept being erased. The model learns from it's own knowledge to steer the diffusion process away from the undesired concept.
<div align='center'>
<img src = 'images/ESD.png'>
</div>

## Generating Images

Generating images from custom ESD model is super easy. Please follow `inference.ipynb` notebook

### UPDATE (NudeNet)
If you want to recreate the results from our paper on NSFW task - please use this https://drive.google.com/file/d/1J_O-yZMabmS9gBA2qsFSrmFoCl7tbFgK/view?usp=sharing

* Untar this file and save it in the homedirectory '~/.NudeNet'
* This should enable the right results as we use this checkpoint for our analysis.

## Running Gradio Demo Locally

To run the gradio interactive demo locally, clone the files from [demo repository](https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/tree/main) <br>

* Create an environment using the packages included in the requirements.txt file
* Run `python app.py`
* Open the application in browser at `http://127.0.0.1:7860/`
* Train, evaluate, and save models using our method
  
## Citing our work
The preprint can be cited as follows
```
@inproceedings{gandikota2023erasing,
  title={Erasing Concepts from Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Jaden Fiotto-Kaufman and David Bau},
  booktitle={Proceedings of the 2023 IEEE International Conference on Computer Vision},
  year={2023}
}
```
