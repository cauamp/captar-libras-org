import os

from argparse import ArgumentParser
from colorama import Fore


def make_parser():
    parser     = ArgumentParser(description="Define the hyperparameters")

    parser.add_argument('--dataset-location', '-d', type=str, required=True,
                        help="The folder containing the 3 splits of the dataset: train, test and dev.")


    parser.add_argument('--save-location', '-s', type=str, required=False,
                        help="Where the post-processed images will be saved to. Defaults to ./results/")


    parser.set_defaults(save_location="./results")

    return parser


'''
Just makes sure some parameters are correct. 
It does NOT cover all use cases. Be smart about what parameters you're passing over. 

Don't fuck shit up on purpose!!!
'''
def validateParams(args):

    # Checks if args.dataset_location contains a train, test AND dev folder
    for subset in ["train", "test", "dev"]:
        if subset not in os.listdir(args.dataset_location):
            raise FileNotFoundError(f"{Fore.RED}(--dataset-location):  Dataset folder {Fore.GREEN} {args.dataset_location} {Fore.RED} Does not contain a train, test and dev subset. Are you sure this is correct?{Fore.WHITE}")
    
    if not os.path.isdir(args.save_location):
        os.makedirs(args.save_location)

        [ os.makedirs(f"{args.save_location}/{subset}") for subset in ["train", "test", "dev"] ]