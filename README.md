# data_driven_convex_regularization
This repo contains python scripts for implementating data-driven convex regularization for inverse problems (sparse-view CT reconstruction, in particular). For a detailed description of the algorithm and theoretical results, see: https://arxiv.org/abs/2008.02839.

If you use these scripts in your research, consider citing the paper:
```
@misc{mukherjee2021learned,
      title={Learned convex regularizers for inverse problems}, 
      author={Subhadip Mukherjee and Sören Dittmer and Zakhar Shumaylov and Sebastian Lunz and Ozan Öktem and Carola-Bibiane Schönlieb},
      year={2021},
      eprint={2008.02839},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
# Steps to run the scripts:
* The phantoms used in our CT experiments are available (as `.npy` files) here: https://drive.google.com/drive/folders/1SHN-yti3MgLmmW_l0agZRzMVtp0kx6dD?usp=sharing. Download the `.zip` file containing the phantoms, unzip, and put inside the cloned directory.
* Create a conda environment with the required dependencies by `conda env create -f environment.yml`, and then activate it by `conda activate env_deep_learning`. 
* Run `python simulate_projections_for_train_and_test.py` to simulate the projection data and the FBP solutions. 
* Train a convex regularizer by `python train_convex_reg.py`. 
* Evaluate the model on test slices by running `python eval_convex_reg.py`. 
* If you want to test the model for a different acquisition geometry, appropriately midify the acquisition parameters in `simulate_projections_for_train_and_test.py`.  
