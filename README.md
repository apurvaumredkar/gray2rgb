# UB CSE 573 Computer Vision and Image Processing - Capstone Project
Topic: Auto-colorization of grayscale images using deep learning.
---
Team members:
- Saksham Lakhera
- Apurva Umredkar
- Sharan Raj Sivakumar
---

🚀 Our model is live! Check out it's capability in real time: https://huggingface.co/spaces/TeamSAS/ImageColorizer

In this project, we aim to achieve perceptually accurate colorization of grayscale images by predicting the **ab** channels using the **L** channel as the input to the model (CIELAB color space).
The model leverages a hybrid UNet-CNN & ViT-Tiny model trained adversarially against a Patch GAN Discriminator.

![Generator Diagram](https://github.com/user-attachments/assets/2b3bc6b3-673b-4b47-be0a-c702dbd52c54)

Link to original dataset: http://places2.csail.mit.edu/download.html

Link to our data subset used for training: https://drive.google.com/file/d/1ubhq38S2fr-HNQL6-uosoQ0PqUKlM9Lj/view?usp=sharing

NOTE: The `lab_weights.npz` file in this repository contains color channel distribution weights for the above subset. Generating a new datasubset using `generate_data_subset.py` will generate new weights.

### Instructions to run the pipeline:
- Set the hyperparameters in `hyperparameters.json`
- Execute `main.py`. Data subset generation will be skipped if the folder already exists. Run with flag `--generate_data` to force regeneration.
- If you wish to run evaluation with a specific model checkpoint, execute `python main.py --evaluate --checkpoint "<path to checkpoint>"` or `python evaluate.py --checkpoint "<path to checkpoint>"`
- Run all cells in the `visualize_results.ipynb` file to see training & validation history trends, & model outputs against ground truth.
