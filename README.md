# VAE-face-generation
Train a VAE model based on CelebA dataset. Model can generate random face images and generate face transformation between two given images.
---
## Folder Structure:
  ../imgs (output images go here)
  ../logs (output TB logs go here)
  ../models (trained models go here)
  ../samples/celeba/ (CelebA dataset in here)
  ../samples/photos/ (two images for face transformation go here)
---
## To Run:
  In console:
    python VAE_face_generation.py (training, face generation, face transformation)
    python VAE_face_generation.py --call test (face generation)
    python VAE_face_generation.py --call transfer (face transformation)
