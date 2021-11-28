import os
import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
from encoder.generator_model import Generator


def move_latent_and_save(generator, latent_vector, direction, coeffs):
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]

        new_latent_vector = new_latent_vector.reshape((1, 18, 512))
        generator.set_dlatents(new_latent_vector)
        img_array = generator.generate_images()[0]
        img = PIL.Image.fromarray(img_array, 'RGB')
        img.save('results/result'+str(i).zfill(3)+'.png')

def main():

    ## load model 

    tflib.init_tf()

    local_file = "models/generator_yellow.pkl"
    #local_file = "models/karras2019stylegan-ffhq-1024x1024.pkl"
    f = open(local_file, 'rb')
    with f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

    # Loading already learned representations
    target1 = np.load('latent_representations/p2_01.npy')

    # Of course you can learn your own vectors using two scripts
    # 1) Extract and align faces from images
    # python align_images.py raw_images/ aligned_images/
    # 2) Find latent representation of aligned images
    # python encode_images.py aligned_images/ generated_images/ latent_representations/

    # Loading already learned latent directions
    smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')
    gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
    age_direction = np.load('ffhq_dataset/latent_directions/age.npy')

    # In general it's possible to find directions of almost any face attributes: position, hair style or color ... 
    # Additional scripts for doing so will be realised soon
    move_latent_and_save(generator, target1, smile_direction, [-5, -2, -1, 0, 1, 2, 5])

if __name__ == "__main__":
    main()
