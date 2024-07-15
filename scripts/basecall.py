"""
Script used for basecalling

Adapted from https://github.com/marcpaga/basecalling_architectures/blob/5db4957496079d19deacb01c9f4f4957f7257f49/scripts/basecall_original.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from basecaller import BasecallerCRF
from dataset import BaseFast5Dataset
from model import MaskedModel as Model

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print("Starting")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast5-dir", type=str, help='Path to fast5 files', default = None)
    parser.add_argument("--fast5-list", type=str, help='Path to file with list of files to be processed', default = None)
    parser.add_argument("--checkpoint", type=str, help='checkpoint file to load model weights', required = True)
    parser.add_argument("--output-file", type=str, help='output fastq file', required = True)
    parser.add_argument("--chunk-size", type=int, default = 2000)
    parser.add_argument("--window-overlap", type=int, default = 200)
    parser.add_argument("--batch-size", type=int, default = 64)
    parser.add_argument("--beam-size", type=int, default = 1)
    parser.add_argument("--beam-threshold", type=float, default = 0.1)
    parser.add_argument("--model-stride", type=int, default = None)
    
    args = parser.parse_args()


    file_list = list()
    if args.fast5_dir is not None:
        if args.fast5_dir.endswith('.fast5'):
            file_list.append(args.fast5_dir)
        else:
            for f in os.listdir(args.fast5_dir):
                if f.endswith('.fast5'):
                    file_list.append(os.path.join(args.fast5_dir, f))
    elif args.fast5_list is not None:
        with open(args.fast5_list, 'r') as f:
            for line in f:
                file_list.append(line.strip('\n'))
    else:
        raise ValueError('Either --fast5-dir or --fast5-list must be given')

    print('Found ' + str(len(file_list)) + ' files')

    fast5_dataset = BaseFast5Dataset(fast5_list= file_list, buffer_size = 1)

    # load model
    checkpoint_file = args.checkpoint

    model = Model(load_default=True, device=device, dataloader_train=None, dataloader_validation=None)
    model = model.to(device)

    model.load(checkpoint_file, initialize_lazy = True)

    model = model.to(device)

    basecaller = BasecallerCRF(
        dataset=fast5_dataset, model=model, batch_size=args.batch_size,
        output_file=args.output_file, n_cores=4, chunksize=args.chunk_size,
        overlap=args.window_overlap, stride=args.model_stride,
        beam_size=args.beam_size, beam_threshold=args.beam_threshold,
    )

    start_time = time.time()

    basecaller.basecall(verbose = True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time elapsed: {:.1f} s ({:.1f} files/second)'.format(elapsed_time, len(file_list) / elapsed_time))
