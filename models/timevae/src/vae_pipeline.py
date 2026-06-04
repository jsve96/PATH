import os
import numpy as np
from .data_utils import (
    load_yaml_file,
    split_data,
)

from .vae.vae_utils import (
    instantiate_vae_model,
    train_vae,
    save_vae_model,
    get_posterior_samples,
    get_prior_samples,
    load_vae_model,
)


def train_vae_pipeline(data: np.array, vae_type: str, ckpt_path : str, vae_epochs: int):
    # ----------------------------------------------------------------------------------
   
    # split train data into train/valid splits
    train_data, valid_data = split_data(data, valid_perc=0.1, shuffle=True)

    # already scaled (scale data) 
    #scaled_train_data, scaled_valid_data, scaler = scale_data(train_data, valid_data)

    # ----------------------------------------------------------------------------------
    # Instantiate and train the VAE Model
    # load hyperparameters from yaml file
    hyperparameters = load_yaml_file("./models/timevae/src/config/hyperparameters.yaml")[vae_type]

    # instantiate the model
    _, sequence_length, feature_dim = train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )

    # train vae
    train_vae(
        vae=vae_model,
        train_data=train_data,
        max_epochs=vae_epochs, ##change back
        verbose=1,
    )

    
    save_vae_model(vae=vae_model, dir_path=ckpt_path)

    
    x_decoded = get_posterior_samples(vae_model, train_data)
    

    # Generate prior samples
    ## This is only for sampling mode
    prior_samples = get_prior_samples(vae_model, num_samples=train_data.shape[0])
    

    # inverse transformer samples to original scale and save to dir
    # save_data(
    #     data=prior_samples,
    #     output_file=F"{ckpt_path}/samples.npy"
    #     )
    print(f"Train Samples saved at {ckpt_path}/samples.npy")
    np.save(f"{ckpt_path}/samples.npy",prior_samples)


    # ----------------------------------------------------------------------------------
    

    # ----------------------------------------------------------------------------------
    # # later.... load model
    # loaded_model = load_vae_model(vae_type, model_save_dir).to(next(vae_model.parameters()).device)

    # # Verify that loaded model produces same posterior samples
    # new_x_decoded = loaded_model.predict(scaled_train_data)
    # print(
    #     "Preds from orig and loaded models equal: ",
    #     np.allclose(x_decoded, new_x_decoded, atol=1e-5),
    # )

    # ----------------------------------------------------------------------------------



def sample_vae_pipeline(train_data: np.array, n_samples: int ,vae_type:str, ckpt_path : str):

    hyperparameters = load_yaml_file("./models/timevae/src/config/hyperparameters.yaml")[vae_type]

    # instantiate the model
    _, sequence_length, feature_dim = train_data.shape
    vae_model = instantiate_vae_model(
        vae_type=vae_type,
        sequence_length=sequence_length,
        feature_dim=feature_dim,
        **hyperparameters,
    )

    loaded_model = load_vae_model(vae_type, ckpt_path).to(next(vae_model.parameters()).device)
    
    samples = get_prior_samples(loaded_model,num_samples=n_samples)

    return samples
# if __name__ == "__main__":
#     # check `/data/` for available datasets
#     dataset = "sine_subsampled_train_perc_20"

#     # models: vae_dense, vae_conv, timeVAE
#     model_name = "vae_conv"

#     run_vae_pipeline(dataset, model_name)
