# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import argparse
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib.tflib as tflib

def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Use trained model to generate images")
    parser.add_argument("--model_checkpoint", required=True, type=str, help="Configuration file")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Output directory that will contain histograms and CSV file with results")
    args = parser.parse_args()
    return args

def main():
    args = do_parsing()
    print(args)

    tflib.init_tf()
    _G, _D, Gs = pickle.load(open(args.model_checkpoint, "rb"))
    Gs.print_layers()

    for i in range(0,25):
        rnd = np.random.RandomState(None)
        latents = rnd.randn(1, Gs.input_shape[1])
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.6, randomize_noise=True, output_transform=fmt)
        os.makedirs(args.output_dir, exist_ok=True)
        png_filename = os.path.join(args.output_dir, 'example-'+str(i)+'.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
