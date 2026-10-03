import torch
import ctcdecode
import numpy as np
from itertools import groupby


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(vocab, beam_width=10, blank_id=blank_id,
                                                    num_processes=10)

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        '''
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        '''
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(nn_output, vid_lgt)
        ret_list = []
        probabilities = []  # List to store the probabilities for each class in the decoded result in each batch
        beam_confidences = []  # To store the beam scores (confidences for the chosen sequence)
        
        for batch_idx, (output, beam_seq, seq_len, score) in enumerate(zip(nn_output, beam_result, out_seq_len, beam_scores)):
                                           
            # Decode the first result in the beam
            first_result = beam_seq[0][:seq_len[0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            
            # Map gloss IDs to their corresponding gloss names
            decoded_result = [
                (self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in enumerate(first_result)
            ]
            ret_list.append(decoded_result)

            # Map probabilities for each class in the decoded result
            class_probabilities = [
                output[idx, gloss_id.item()].item()  # Probability for each class
                for idx, gloss_id in enumerate(first_result)
            ]
            probabilities.append(class_probabilities)

            # Extract the beam score and compute confidence
            beam_score = score[0].item()  # Log score for the first beam
            beam_confidence = 1 / np.exp(beam_score)  # Convert log score to probability
            beam_confidences.append(beam_confidence)

        return ret_list, probabilities, beam_confidences

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:vid_lgt[batch_idx]])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
        return ret_list
