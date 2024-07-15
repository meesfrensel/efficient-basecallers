"""
Basecaller classes

Copied from https://github.com/marcpaga/basecalling_architectures/blob/5db4957496079d19deacb01c9f4f4957f7257f49/src/classes.py
"""

import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import uuid

from constants import STICH_ALIGN_FUNCTION, STICH_GAP_OPEN_PENALTY, STICH_GAP_EXTEND_PENALTY, MATRIX
from dataset import BaseFast5Dataset
from evaluation import make_align_arr, elongate_cigar

class BaseBasecaller():

    def __init__(self, dataset, model, batch_size, output_file, n_cores = 4, chunksize = 2000, overlap = 400, stride = None, beam_size = 1, beam_threshold = 0.1):

        assert isinstance(dataset, BaseFast5Dataset)

        self.dataset = DataLoader(dataset, batch_size=1, shuffle=False, num_workers = 4)
        self.model = model
        self.batch_size = batch_size
        self.output_file = output_file
        self.n_cores = n_cores
        self.chunksize = chunksize
        self.overlap = overlap
        if stride is None:
            self.stride = self.model.cnn_stride
        else:
            self.stride = stride
        self.beam_size = beam_size
        self.beam_threshold = beam_threshold

    def stitch(self, method, *args, **kwargs):
        """
        Stitch chunks together with a given overlap
        
        Args:
            chunks (tensor): predictions with shape [samples, length, classes]
        """

        if method == 'stride':
            return self.stitch_by_stride(*args, **kwargs)
        elif method == 'alignment':
            return self.stitch_by_alignment(*args, **kwargs)
        else:
            raise NotImplementedError()

    def basecall(self, verbose = True):
        raise NotImplementedError()
    
    def stitch_by_stride(self, chunks, chunksize, overlap, length, stride, reverse=False):
        """
        Stitch chunks together with a given overlap
        
        This works by calculating what the overlap should be between two outputed
        chunks from the network based on the stride and overlap of the inital chunks.
        The overlap section is divided in half and the outer parts of the overlap
        are discarded and the chunks are concatenated. There is no alignment.
        
        Chunk1: AAAAAAAAAAAAAABBBBBCCCCC
        Chunk2:               DDDDDEEEEEFFFFFFFFFFFFFF
        Result: AAAAAAAAAAAAAABBBBBEEEEEFFFFFFFFFFFFFF
        
        Args:
            chunks (tensor): predictions with shape [samples, length, *]
            chunk_size (int): initial size of the chunks
            overlap (int): initial overlap of the chunks
            length (int): original length of the signal
            stride (int): stride of the model
            reverse (bool): if the chunks are in reverse order
            
        Copied from https://github.com/nanoporetech/bonito
        """

        if isinstance(chunks, np.ndarray):
            chunks = torch.from_numpy(chunks)

        if chunks.shape[0] == 1: return chunks.squeeze(0)

        semi_overlap = overlap // 2
        start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
        stub = (length - overlap) % (chunksize - overlap)
        first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

        if reverse:
            chunks = list(chunks)
            return torch.cat([
                chunks[-1][:-start], *(x[-end:-start] for x in reversed(chunks[1:-1])), chunks[0][-first_chunk_end:]
            ])
        else:
            return torch.cat([
                chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
            ])

    def stitch_by_alignment(self, preds, qscores_list, num_patch_bases = 10):

        consensus = list()
        phredq_consensus = list()
        for i in range(0, len(preds) - 2, 2):
            
            if i == 0:
                ref1 = preds[i]
                ref1_phredq = qscores_list[i]

            ref2 = preds[i+2]
            ref2_phredq = qscores_list[i+2]
            que = preds[i+1]
            que_phredq = qscores_list[i+1]

            alignment = STICH_ALIGN_FUNCTION(que, ref1+ref2, open = STICH_GAP_OPEN_PENALTY, extend = STICH_GAP_EXTEND_PENALTY, matrix = MATRIX)

            decoded_cigar = alignment.cigar.decode.decode()
            long_cigar, _, _ = elongate_cigar(decoded_cigar)
            align_arr = make_align_arr(long_cigar, ref1+ref2, que, phredq = que_phredq, phredq_ref = ref1_phredq+ref2_phredq)


            n_gaps = 0
            st_first_segment = len(ref1) - num_patch_bases
            while True:
                n_gaps_new = np.sum(align_arr[0][:st_first_segment] == '-')
                if n_gaps_new > n_gaps:
                    st_first_segment += n_gaps_new
                    n_gaps = n_gaps_new
                else:
                    break

            n_gaps = 0
            nd_first_segment = st_first_segment + num_patch_bases
            while True:
                n_gaps_new = np.sum(align_arr[0][st_first_segment:nd_first_segment] == '-')
                if n_gaps_new > n_gaps:
                    nd_first_segment += n_gaps_new
                    n_gaps = n_gaps_new
                else:
                    break

            st_second_segment = nd_first_segment
            nd_second_segment = st_second_segment + num_patch_bases

            n_gaps = 0
            while True:
                n_gaps_new = np.sum(align_arr[0][st_second_segment:nd_second_segment] == '-')
                if n_gaps_new > n_gaps:
                    nd_second_segment += n_gaps_new
                    n_gaps = n_gaps_new
                else:
                    break

            segment1_patch = "".join(align_arr[2][st_first_segment:nd_first_segment].tolist()).replace('-', '')
            segment2_patch = "".join(align_arr[2][st_second_segment:nd_second_segment].tolist()).replace('-', '')
            segment1_patch_phredq = "".join(align_arr[3][st_first_segment:nd_first_segment].tolist()).replace(' ', '')
            segment2_patch_phredq = "".join(align_arr[3][st_second_segment:nd_second_segment].tolist()).replace(' ', '')

            new_ref1 = ref1[:-num_patch_bases] + segment1_patch
            new_ref1_phredq = ref1_phredq[:-num_patch_bases] + segment1_patch_phredq
            ref1 = segment2_patch + ref2[num_patch_bases:] 
            ref1_phredq = segment2_patch_phredq + ref2_phredq[num_patch_bases:] 
            assert len(new_ref1) == len(new_ref1_phredq)

            consensus.append(new_ref1)
            phredq_consensus.append(new_ref1_phredq)

        return "".join(consensus), "".join(phredq_consensus), '+'  

class BasecallerCRF(BaseBasecaller):

    def __init__(self, *args, **kwargs):
        super(BasecallerCRF, self).__init__(*args, **kwargs)


    def basecall(self, verbose = True, qscale = 1.0, qbias = 1.0):
        # iterate over the data

        assert self.dataset.dataset.buffer_size == 1
        
        for batch in tqdm(self.dataset, disable = not verbose):

            ids = batch['id'].squeeze(0)
            ids_arr = np.zeros((ids.shape[0], ), dtype = 'U36')
            for i in range(ids.shape[0]):
                ids_arr[i] = str(uuid.UUID(fields=ids[i].tolist()))

            assert len(np.unique(ids_arr)) == 1
            read_id = np.unique(ids_arr)[0]
            filepath = bytes(batch['file'].squeeze(0)[0].tolist()).decode()
            read_file = Path(filepath).stem
            
            x = batch['x'].squeeze(0)
            l = x.shape[0]
            ss = torch.arange(0, l, self.batch_size)
            nn = ss + self.batch_size

            transition_scores = list()
            for s, n in zip(ss, nn):
                p = self.model.predict_step({'x':x[s:n, :]})
                scores = self.model.compute_scores(p, use_fastctc=True)
                transition_scores.append(scores[0].cpu())
            init = scores[1][0, 0].cpu()

            stacked_transitions = self.stitch_by_stride(
                chunks = np.vstack(transition_scores), 
                chunksize = self.chunksize, 
                overlap = self.overlap, 
                length = batch['len'].squeeze(0)[0].item(), 
                stride = self.stride
            )


            if self.beam_size == 1:
                seq, path = self.model._decode_crf_greedy_fastctc(
                    tracebacks = stacked_transitions.numpy(), 
                    init = init.numpy(), 
                    qstring = True, 
                    qscale = qscale, 
                    qbias = qbias,
                    return_path = True
                )

                fastq_string = '@'+str(read_id)+'\n'
                # fastq_string = '@'+str(read_file)+'\n'
                fastq_string += seq[:len(path)] + '\n'
                fastq_string += '+\n'
                fastq_string += seq[len(path):] + '\n'
                
            else:
                seq = self.model._decode_crf_beamsearch_fastctc(
                    tracebacks = stacked_transitions.numpy(), 
                    init = init.numpy(), 
                    beam_size = self.beam_size, 
                    beam_cut_threshold = self.beam_threshold, 
                    return_path = False
                )

                fastq_string = '@'+str(read_id)+'\n'
                # fastq_string = '@'+str(read_file)+'\n'
                fastq_string += seq + '\n'
                fastq_string += '+\n'
                fastq_string += '?'*len(seq) + '\n'
            
            with open(self.output_file, 'a') as f:
                f.write(str(fastq_string))
                f.flush()
