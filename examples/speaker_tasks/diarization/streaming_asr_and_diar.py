#!/usr/bin/env python
# coding: utf-8

import time
import numpy as np
import pyaudio as pa
import argparse
import os
import nemo
import nemo.collections.asr as nemo_asr
import soundfile as sf
from pyannote.metrics.diarization import DiarizationErrorRate
import sklearn.metrics.pairwise as pw
from scipy.io import wavfile
from scipy.optimize import linear_sum_assignment
import librosa
import ipdb
from datetime import datetime

### From speaker_diarize.py
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from nemo.collections.asr.parts.utils.streaming_utils import longest_common_subsequence_merge as lcs_alg
from nemo.collections.asr.models.classification_models import EncDecClassificationModel
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import FrameBatchASR_Logits, WERBPE_TS, ASR_TIMESTAMPS, WER_TS
from nemo.collections.asr.data.audio_to_label import repeat_signal
from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE, write_txt
from nemo.collections.asr.parts.utils.speaker_utils import audio_rttm_map
from nemo.collections.asr.parts.utils.speaker_utils import get_contiguous_stamps, merge_stamps, labels_to_pyannote_object, rttm_to_labels, labels_to_rttmfile, get_uniqname_from_filepath, get_embs_and_timestamps
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    get_vad_stream_status,
    prepare_manifest,
)
from typing import Dict, List, Tuple, Type, Union
from nemo.collections.asr.models import ClusteringDiarizer, EncDecCTCModel, EncDecCTCModelBPE
from sklearn.preprocessing import OneHotEncoder
from nemo.collections.asr.parts.utils.streaming_utils import AudioFeatureIterator, FrameBatchASR
from nemo.collections.asr.parts.utils.nmse_clustering import (
    NMESC,
    _SpectralClustering,
    getEnhancedSpeakerCount,
    COSclustering,
    getCosAffinityMatrix,
    getAffinityGraphMat,
)
from nemo.collections.asr.parts.utils.nmesc_clustering import (
    NMESC,
    SpectralClustering,
    getEnhancedSpeakerCount,
    COSclustering,
    getCosAffinityMatrix,
    getAffinityGraphMat,
    getMultiScaleCosAffinityMatrix
)
from nemo.collections.asr.parts.utils.nmesc_clustering import COSclustering

from nemo.core.config import hydra_runner
from nemo.utils import logging
import hydra
from typing import List, Optional, Dict
from omegaconf import DictConfig, OmegaConf, open_dict
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
import copy
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from nemo.utils import logging, model_utils
import torch
from torch.utils.data import DataLoader
import math

from collections import Counter
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# For streaming ASR
from nemo.core.classes import IterableDataset
from torch.utils.data import DataLoader
import math
import difflib
from sklearn.manifold import TSNE
from nemo.core.classes import IterableDataset


TOKEN_OFFSET = 100

import contextlib
import json
import os

import editdistance
from sklearn.model_selection import ParameterGrid

import nemo
import nemo.collections.asr as nemo_asr

from nemo.utils import logging

# import scripts.asr_language_modeling.ngram_lm.kenlm_utils as kenlm_utils
# from ctcdecode import OnlineCTCBeamDecoder, CTCBeamDecoder
from pyctcdecode import build_ctcdecoder

seed_everything(42)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            # logging.info('%2.2fms %r'%((te - ts) * 1000, method.__name__))
            pass
        return result
    return timed


def isOverlap(rangeA, rangeB):
    start1, end1 = rangeA
    start2, end2 = rangeB
    return end1 > start2 and end2 > start1

def getOverlapRange(rangeA, rangeB):
    assert isOverlap(rangeA, rangeB)
    return [ max(rangeA[0], rangeB[0]), min(rangeA[1], rangeB[1])]

def getOverlapAmount(rangeA, rangeB):
    start1, end1 = rangeA
    start2, end2 = rangeB
    return max(max((end2-start1), 0) - max((end2-end1), 0) - max((start2-start1), 0), 0)

def combine_overlaps(ranges):
    return reduce(
        lambda acc, el: acc[:-1:] + [(min(*acc[-1], *el), max(*acc[-1], *el))]
            if acc[-1][1] >= el[0] - 1
            else acc + [el],
        ranges[1::],
        ranges[0:1],
    )

def getMergedRanges(label_list_A, label_list_B):
    if label_list_A == [] and label_list_B != []:
        return label_list_B
    elif label_list_A != [] and label_list_B == []:
        return label_list_A
    else:
        label_list_A = [ [fl2int(x[0]), fl2int(x[1])] for x in label_list_A] 
        label_list_B = [ [fl2int(x[0]), fl2int(x[1])] for x in label_list_B] 

        combined = combine_overlaps(label_list_A + label_list_B)

        return [ [int2fl(x[0]), int2fl(x[1])] for x in combined ]

def getSubRangeList(target_range: List, source_list: List) -> List:
    if target_range == []:
        return []
    else:
        out_range_list = []
        for s_range in source_list:
            if isOverlap(s_range, target_range):
                ovl_range = getOverlapRange(s_range, target_range)
                out_range_list.append(ovl_range)
        return out_range_list 

def fl2int(x):
    return int(x*100)

def int2fl(x):
    return round(float(x/100.0), 2)

def get_partial_ref_labels(pred_labels, ref_labels):
    last_pred_time = float(pred_labels[-1].split()[1])
    ref_labels_out = []
    for label in ref_labels:
        start, end, speaker = label.split()
        start, end = float(start), float(end)
        if last_pred_time <= start:
            pass
        elif start < last_pred_time <= end:
            label = f"{start} {last_pred_time} {speaker}"
            ref_labels_out.append(label) 
        elif end < last_pred_time:
            ref_labels_out.append(label) 
    return ref_labels_out 

def speech_collate_fn(batch):
    """collate batch of audio sig, audio len
    Args:
        batch (FloatTensor, LongTensor):  A tuple of tuples of signal, signal lengths.
        This collate func assumes the signals are 1d torch tensors (i.e. mono audio).
    """

    _, audio_lengths = zip(*batch)

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
   
    
    audio_signal= []
    for sig, sig_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        
    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    return audio_signal, audio_lengths

def get_DER(all_reference, all_hypothesis, collar=0.5, skip_overlap=True):
    """
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap, uem=None)

    mapping_dict = {}
    for k, (reference, hypothesis) in enumerate(zip(all_reference, all_hypothesis)):
        metric(reference, hypothesis, detailed=True)
        mapping_dict[k] = metric.optimal_mapping(reference, hypothesis)

    DER = abs(metric)
    CER = metric['confusion'] / metric['total']
    FA = metric['false alarm'] / metric['total']
    MISS = metric['missed detection'] / metric['total']

    metric.reset()

    return DER, CER, FA, MISS, mapping_dict

def get_wer_feat_logit_single(samples, frame_asr, frame_len, tokens_per_chunk, delay, model_stride_in_secs, frame_mask):
    """
    Create a preprocessor to convert audio samples into raw features,
    Normalization will be done per buffer in frame_bufferer.
    """
    hyps, tokens_list = [], []
    frame_asr.reset()
    feature_frame_shape = frame_asr.read_audio_file_and_return_samples(samples, delay, model_stride_in_secs, frame_mask)
    if frame_mask is not None:
        hyp, tokens, _ = frame_asr.transcribe_with_ts(tokens_per_chunk, delay)
    else:
        hyp, tokens = None, None
    hyps.append(hyp)
    tokens_list.append(tokens)
    return hyps, tokens_list, feature_frame_shape

class StreamFeatureIterator(AudioFeatureIterator):
    def __init__(self, samples, frame_len, preprocessor, device, frame_mask=None):
        super().__init__(samples, frame_len, preprocessor, device)
        if frame_mask is not None:
            self._features = torch.log(torch.mul(np.exp(1) ** self._features, frame_mask.to(device)))

class FrameBatchASR_Logits_Sample(FrameBatchASR_Logits):
    """
    A class for streaming frame-based ASR.
    Inherits from FrameBatchASR and adds new capability of returning the logit output.
    Please refer to FrameBatchASR for more detailed information.
    """

    def __init__(self, asr_model, frame_len=1.0, total_buffer=4.0, batch_size=4):
        super().__init__(asr_model, frame_len, total_buffer, batch_size)
    
    @timeit
    def read_audio_file_and_return_samples(self, _samples, delay: float, model_stride_in_secs: float, frame_mask):
        self.device = self.asr_model.device
        samples = np.pad(_samples, (0, int(delay * model_stride_in_secs * self.asr_model._cfg.sample_rate)))
        frame_reader = StreamFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.device, frame_mask)
        self.set_frame_reader(frame_reader)
        return frame_reader._features.shape
        
        # ########## 
        # if asr_diar.frame_index == 0:
            # self.frame_reader = StreamFeatureIterator(samples, self.frame_len, self.raw_preprocessor, self.device)
            # # self.prev_feats = [copy.deepcopy(self.frame_reader._features_buffer), copy.deepcopy(self.frame_reader._features_len)]
            # self.set_frame_reader(self.frame_reader)
        # else:
            # audio_signal = torch.from_numpy(new_samples).unsqueeze_(0).to(self.device)
            # audio_signal_len = torch.Tensor([new_samples.shape[0]]).to(self.device)
            # feat_frame, feats_len = self.raw_preprocessor(input_signal=audio_signal, length=audio_signal_len,)
            # feat_frame = feat_frame.squeeze()
            # import ipdb; ipdb.set_trace()
            # # _prev_feats = copy.deepcopy(self.prev_feats)
            # # self.prev_feats[0][:, : -n_frame_len] = self.prev_feats[0][:, n_frame_len:]
            # # self.prev_feats[0][:, -feat_frame.shape[1]:] = feat_frame
            # # self.frame_reader = StreamFeatureIterator(None, self.frame_len, self.raw_preprocessor, self.device, self.prev_feats)
            # # self.set_frame_reader(self.frame_reader)
            # # self.frame_bufferer.get_frame_buffers(feat_frame.cpu().numpy()[1:])


def callback_sim(asr_diar, uniq_id, buffer_counter, sdata, frame_count, time_info, status):
    start_time = time.time()
    asr_diar.buffer_counter = buffer_counter
    sampled_seg_sig = sdata[asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter):asr_diar.CHUNK_SIZE*(asr_diar.buffer_counter+1)]
    asr_diar.uniq_id = uniq_id
    asr_diar.signal = sdata
    words, timestamps, pred_diar_labels = asr_diar.transcribe(sampled_seg_sig)
    print("words: ", words)
    print("timestamps: ", timestamps)
    if asr_diar.buffer_start >= 0 and (pred_diar_labels != [] and pred_diar_labels != None):
        asr_diar._update_word_and_word_ts(words, timestamps)
        string_out = asr_diar._get_speaker_label_per_word(uniq_id, asr_diar.word_seq, asr_diar.word_ts_seq, pred_diar_labels)
        write_txt(f"{asr_diar.diar._out_dir}/print_script.sh", string_out.strip())
        
    ETA = time.time()-start_time 
    if asr_diar.diar.params['force_real_time']:
        assert ETA < asr_diar.frame_len, "The process has failed to be run in real-time."
        time.sleep(1.0 - ETA*1.0)

class OnlineClusteringDiarizer(ClusteringDiarizer, ASR_DIAR_OFFLINE):
# class OnlineClusteringDiarizer(ClusteringDiarizer):
    def __init__(self, cfg: DictConfig, params: Dict):
        super().__init__(cfg)
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # Convert config to support Hydra 1.0+ instantiation
        self.uniq_id = get_uniqname_from_filepath(params['single_audio_file_path'])
        cfg = model_utils.maybe_update_config_version(cfg)
        self._cfg = cfg
        self.device =  cfg.diarizer.device
        self.params = params
        self._out_dir = self._cfg.diarizer.out_dir
        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)
        self.AUDIO_RTTM_MAP = None
        
        self.embs_array = {self.uniq_id : {} }
        self.time_stamps = {self.uniq_id : {} }
        self.segment_abs_time_range_list = {self.uniq_id: {} }
        self.segment_raw_audio_list = {self.uniq_id: {} }
        for scale_idx, (window, shift) in self.multiscale_args_dict['scale_dict'].items():
            self.multiscale_embeddings_and_timestamps[scale_idx] = [None, None]
            self.embs_array[self.uniq_id][scale_idx] = None
            self.time_stamps[self.uniq_id][scale_idx] = []
            self.segment_abs_time_range_list[self.uniq_id][scale_idx] = []
            self.segment_raw_audio_list[self.uniq_id][scale_idx] = []
        self.base_scale_index = max(self.multiscale_args_dict['scale_dict'].keys())
        
        torch.manual_seed(0)
        self._speaker_model.to(self.device)
        self._speaker_model.eval()
        self.paths2session_audio_files = []
        self.all_hypothesis = []
        self.all_reference = []
        self.out_rttm_dir = None

        self.ROUND = 2
        self.embed_seg_len = self._cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec
        self.embed_seg_hop = self._cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec
        self.max_num_speakers = 8
        self._current_buffer_segment_count = 64
        self._history_buffer_segment_count = 64
        self.MINIMUM_CLUS_BUFFER_SIZE = 10
        self.MINIMUM_HIST_BUFFER_SIZE = 32
        self._minimum_segments_per_buffer = int(self._history_buffer_segment_count/self.max_num_speakers)
        self.fine_segment_abs_time_range_list = []
        self.fine_segment_raw_audio_list = []
        self.Y_fullhist = []
        self.use_online_mat_reduction = True
        # self.segment_abs_time_range_list = []
        # self.segment_raw_audio_list = []
        self.history_embedding_buffer_emb = np.array([])
        self.history_embedding_buffer_label = np.array([])
        self.history_buffer_seg_start = None
        self.history_buffer_seg_end = None
        self.old_history_buffer_seg_end = None
        self.last_emb_in_length = -float('inf')
        self.frame_index = None
        self.index_dict = {'max_embed_count': 0}
        self.cumulative_speaker_count = {}
        self.embedding_count_history = []
        self.p_value_hist = []

        self.online_diar_buffer_segment_quantity = params['online_history_buffer_segment_quantity']
        self.online_history_buffer_segment_quantity = params['online_diar_buffer_segment_quantity']
        self.enhanced_count_thres = params['enhanced_count_thres']
        self.max_num_speaker = params['max_num_speaker']
        self.oracle_num_speakers = None

        self.diar_eval_count = 0
        self.DER_csv_list = []
        self.der_dict = {}
        self.der_stat_dict = {"avg_DER":0, "avg_CER":0, "max_DER":0, "max_CER":0, "cum_DER":0, "cum_CER":0}
        self.color_palette = {'speaker_0': '\033[1;32m',
                              'speaker_1': '\033[1;34m',
                              'speaker_2': '\033[1;30m',
                              'speaker_3': '\033[1;31m',
                              'speaker_4': '\033[1;35m',
                              'speaker_5': '\033[1;36m',
                              'speaker_6': '\033[1;37m',
                              'speaker_7': '\033[1;30m',
                              'speaker_8': '\033[1;33m',
                              'speaker_9': '\033[0;34m',
                              'white': '\033[0;37m'}
    
    @property 
    def online_diar_buffer_segment_quantity(self, value):
        return self._current_buffer_segment_count

    @online_diar_buffer_segment_quantity.setter
    def online_diar_buffer_segment_quantity(self, value):
        logging.info(f"Setting online diarization buffer to : {value}")
        assert value >= self.MINIMUM_CLUS_BUFFER_SIZE, f"Online diarization clustering buffer should be bigger than {self.MINIMUM_CLUS_BUFFER_SIZE}"
        self._current_buffer_segment_count = value # How many segments we want to use as clustering buffer
    
    @property 
    def online_history_buffer_segment_quantity(self, value):
        return self._current_buffer_segment_count

    @online_history_buffer_segment_quantity.setter
    def online_history_buffer_segment_quantity(self, value):
        logging.info(f"Setting online diarization buffer to : {value}")
        assert value >= self.MINIMUM_HIST_BUFFER_SIZE, f"Online diarization history buffer should be bigger than {self.MINIMUM_HIST_BUFFER_SIZE}"
        self._history_buffer_segment_count = value # How many segments we want to use as history buffer

    def getMergeQuantity(self, new_emb_n, before_cluster_labels):
        """
        Determine which embeddings we need to reduce or merge in history buffer.
        We want to merge or remove the embedding in the bigger cluster first.
        At the same time, we keep the minimum number of embedding per cluster
        with the variable named self._minimum_segments_per_buffer.
        The while loop creates a numpy array emb_n_per_cluster.
        that tells us how many embeddings we should remove/merge per cluster.

        Args:
            new_emb_n: (int)
                the quantity of the newly obtained embedding from the stream.

            before_cluster_labels: (np.array)
                the speaker labels of (the history_embedding_buffer_emb) + (the new embeddings to be added)
        """
        targeted_total_n = new_emb_n
        count_dict = Counter(before_cluster_labels)
        spk_freq_count = np.bincount(before_cluster_labels)
        class_vol = copy.deepcopy(spk_freq_count)
        emb_n_per_cluster = np.zeros_like(class_vol).astype(int)
        arg_max_spk_freq = np.argsort(spk_freq_count)[::-1]
        count = 0
        while np.sum(emb_n_per_cluster) < new_emb_n:
            recurr_idx = np.mod(count, len(count_dict))
            curr_idx = arg_max_spk_freq[recurr_idx]
            margin = (spk_freq_count[curr_idx] - emb_n_per_cluster[curr_idx]) - self._minimum_segments_per_buffer
            if margin > 0:
                target_number = min(margin, new_emb_n)
                emb_n_per_cluster[curr_idx] += target_number
                new_emb_n -= target_number
            count += 1
        assert sum(emb_n_per_cluster) == targeted_total_n, "emb_n_per_cluster does not match with targeted number new_emb_n."
        return emb_n_per_cluster

    def reduce_emb(self, cmat, tick2d, emb_ndx, cluster_labels, method='avg'):
        LI, RI = tick2d[0, :], tick2d[1, :]
        LI_argdx = tick2d[0].argsort()

        if method == 'drop':
            cmat_sym = cmat + cmat.T
            clus_score = np.vstack((np.sum(cmat_sym[LI], axis=1), np.sum(cmat_sym[RI], axis=1)))
            selected_dx = np.argmax(clus_score, axis=0)
            emb_idx = np.choose(selected_dx, tick2d)
            result_emb = emb_ndx[emb_idx, :]
        elif method == 'avg':
            LI, RI = LI[LI_argdx], RI[LI_argdx]
            result_emb = 0.5*(emb_ndx[LI, :] + emb_ndx[RI, :])
        else:
            raise ValueError(f'Method {method} does not exist. Abort.')
        merged_cluster_labels = cluster_labels[np.array(list(set(LI)))]
        bypass_ndx = np.array(list(set(range(emb_ndx.shape[0])) - set(list(LI)+list(RI)) ) )
        if len(bypass_ndx) > 0:
            result_emb = np.vstack((emb_ndx[bypass_ndx], result_emb))  
            merged_cluster_labels = np.hstack((cluster_labels[bypass_ndx], merged_cluster_labels))
        return result_emb, LI, merged_cluster_labels
    

    def reduceEmbedding(self, emb_in, mat):
        history_n, current_n = self._history_buffer_segment_count, self._current_buffer_segment_count
        add_new_emb_to_history = True

        if len(self.history_embedding_buffer_emb) > 0:
            if emb_in.shape[0] <= self.index_dict['max_embed_count']:
                # If the number of embeddings is decreased compared to the last trial,
                # then skip embedding merging.
                add_new_emb_to_history = False
                hist_curr_boundary = self.history_buffer_seg_end
            else:
                # Since there are new embeddings, we push the same amount (new_emb_n) 
                # of old embeddings to the history buffer.
                # We should also update self.history_buffer_seg_end which is a pointer.
                hist_curr_boundary = emb_in.shape[0] - self._current_buffer_segment_count
                _stt = self.history_buffer_seg_end # The old history-current boundary
                _end = hist_curr_boundary # The new history-current boundary
                new_emb_n = _end - _stt
                assert new_emb_n > 0, "new_emb_n cannot be 0 or a negative number."
                update_to_history_emb = emb_in[_stt:_end]
                update_to_history_label = self.Y_fullhist[_stt:_end]
                emb = np.vstack((self.history_embedding_buffer_emb, update_to_history_emb))
                before_cluster_labels = np.hstack((self.history_embedding_buffer_label, update_to_history_label))
                self.history_buffer_seg_end = hist_curr_boundary
        else:
            # This else statement is for the very first diarization loop.
            # This is the very first reduction frame.
            hist_curr_boundary = emb_in.shape[0] - self._current_buffer_segment_count
            new_emb_n = mat.shape[0] - (self._current_buffer_segment_count + self._history_buffer_segment_count)
            emb = emb_in[:hist_curr_boundary]
            before_cluster_labels = self.Y_fullhist[:hist_curr_boundary]
            self.history_buffer_seg_end = hist_curr_boundary
       
        # Update the history/current_buffer boundary cursor
        total_emb, total_cluster_labels = [], []
        
        if add_new_emb_to_history:
            class_target_vol = self.getMergeQuantity(new_emb_n, before_cluster_labels)
            
            # Merge the segments in the history buffer
            for spk_idx, N in enumerate(list(class_target_vol)):
                ndx = np.where(before_cluster_labels == spk_idx)[0]
                if N <= 0:
                    result_emb = emb[ndx]
                    merged_cluster_labels = before_cluster_labels[ndx]
                else:
                    cmat = np.tril(mat[:,ndx][ndx,:])
                    tick2d = self.getIndecesForEmbeddingReduction(cmat, ndx, N)
                    spk_cluster_labels, emb_ndx = before_cluster_labels[ndx], emb[ndx]
                    result_emb, tick_sum, merged_cluster_labels = self.reduce_emb(cmat, tick2d, emb_ndx, spk_cluster_labels, method='avg')
                    assert (ndx.shape[0] - N) == result_emb.shape[0], ipdb.set_trace()
                total_emb.append(result_emb)
                total_cluster_labels.append(merged_cluster_labels)
        
            self.history_embedding_buffer_emb = np.vstack(total_emb)
            self.history_embedding_buffer_label = np.hstack(total_cluster_labels)
            assert self.history_embedding_buffer_emb.shape[0] == history_n, ipdb.set_trace()
        else:
            total_emb.append(self.history_embedding_buffer_emb)
            total_cluster_labels.append(self.history_embedding_buffer_label)

        # Add the current buffer
        total_emb.append(emb_in[hist_curr_boundary:])
        total_cluster_labels.append(self.Y_fullhist[hist_curr_boundary:])

        history_and_current_emb = np.vstack(total_emb)
        history_and_current_labels = np.hstack(total_cluster_labels)
        assert history_and_current_emb.shape[0] <= (history_n + current_n), ipdb.set_trace()
        
        self.last_emb_in_length = emb_in.shape[0]
        return history_and_current_emb, history_and_current_labels, current_n, add_new_emb_to_history
    
    def getIndecesForEmbeddingReduction(self, cmat, ndx, N):
        """
        Get indeces of the embeddings we want to merge or drop.

        Args:
            cmat: (np.array)
            ndx: (np.array)
            N: (int)

        Output:
            tick2d: (numpy.array)
        """
        comb_limit = int(ndx.shape[0]/2)
        assert N <= comb_limit, f" N is {N}: {N} is bigger than comb_limit -{comb_limit}"
        idx2d = np.unravel_index(np.argsort(cmat, axis=None)[::-1], cmat.shape)
        num_of_lower_half = int((cmat.shape[0]**2 - cmat.shape[0])/2)
        idx2d = (idx2d[0][:num_of_lower_half], idx2d[1][:num_of_lower_half])
        cdx, left_set, right_set, total_set = 0, [], [], []
        while len(left_set) <  N and len(right_set) < N:
            Ldx, Rdx = idx2d[0][cdx], idx2d[1][cdx] 
            if (not Ldx in total_set) and (not Rdx in total_set):
                left_set.append(Ldx)
                right_set.append(Rdx)
                total_set = left_set + right_set
            cdx += 1
        tick2d = np.array([left_set, right_set])
        return tick2d
    
    @timeit
    def getReducedMat(self, mat, emb):
        margin_seg_n = mat.shape[0] - (self._current_buffer_segment_count + self._history_buffer_segment_count)
        if margin_seg_n > 0:
            mat = 0.5*(mat + mat.T)
            np.fill_diagonal(mat, 0)
            merged_emb, cluster_labels, current_n, add_new = self.reduceEmbedding(emb, mat)
        else:
            merged_emb = emb
            current_n = self._current_buffer_segment_count
            cluster_labels, add_new = None, True
        self.isOnline = (len(self.history_embedding_buffer_emb) != 0)
        return merged_emb, cluster_labels, add_new
    
    def online_eval_diarization(self, pred_labels, rttm_file, ROUND=2):
        pred_diar_labels, ref_labels_list = [], []
        all_hypotheses, all_references = [], []

        if os.path.exists(rttm_file):
            ref_labels_total = rttm_to_labels(rttm_file)
            ref_labels = get_partial_ref_labels(pred_labels, ref_labels_total)
            reference = labels_to_pyannote_object(ref_labels)
            all_references.append(reference)
        else:
            raise ValueError("No reference RTTM file provided.")

        pred_diar_labels.append(pred_labels)

        self.der_stat_dict['ref_n_spk'] = self.get_num_of_spk_from_labels(ref_labels)
        self.der_stat_dict['est_n_spk'] = self.get_num_of_spk_from_labels(pred_labels)
        hypothesis = labels_to_pyannote_object(pred_labels)
        if ref_labels == [] and pred_labels != []:
            logging.info(
                "Streaming Diar [{}][frame-  {}th  ]:".format(
                    self.uniq_id, self.frame_index
                )
            )
            DER, CER, FA, MISS = 100.0, 0.0, 100.0, 0.0
            der_dict, der_stat_dict = self.get_stat_DER(DER, CER, FA, MISS)
            return der_dict, der_stat_dict
        else:
            all_hypotheses.append(hypothesis)
            try:
                DER, CER, FA, MISS, _= get_DER(all_references, all_hypotheses, collar=0.25, skip_overlap=True)
            except:
                DER, CER, FA, MISS = 100.0, 0.0, 100.0, 0.0
            logging.info(
                "Streaming Diar [{}][frame-    {}th    ]: DER:{:.4f} MISS:{:.4f} FA:{:.4f}, CER:{:.4f}".format(
                    self.uniq_id, self.frame_index, DER, MISS, FA, CER
                )
            )

            der_dict, der_stat_dict = self.get_stat_DER(DER, CER, FA, MISS)
            return der_dict, der_stat_dict
    
    def get_stat_DER(self, DER, CER, FA, MISS):
        der_dict = {"DER": round(100*DER, self.ROUND), 
                    "CER": round(100*CER, self.ROUND), 
                    "FA":  round(100*FA, self.ROUND), 
                    "MISS": round(100*MISS, self.ROUND)}
        self.diar_eval_count += 1
        self.der_stat_dict['cum_DER'] += DER
        self.der_stat_dict['cum_CER'] += CER
        self.der_stat_dict['avg_DER'] = round(100*self.der_stat_dict['cum_DER']/self.diar_eval_count, self.ROUND)
        self.der_stat_dict['avg_CER'] = round(100*self.der_stat_dict['cum_CER']/self.diar_eval_count, self.ROUND)
        self.der_stat_dict['max_DER'] = round(max(der_dict['DER'], self.der_stat_dict['max_DER']), self.ROUND)
        self.der_stat_dict['max_CER'] = round(max(der_dict['CER'], self.der_stat_dict['max_CER']), self.ROUND)
        return der_dict, self.der_stat_dict


    def print_time_colored(self, string_out, speaker, start_point, end_point, params, replace_time=False):
        if params['color']:
            color = self.color_palette[speaker]
        else:
            color = ''

        datetime_offset = 16 * 3600
        if float(start_point) > 3600:
            time_str = "%H:%M:%S.%f"
        else:
            time_str = "%M:%S.%f"
        start_point_str = datetime.fromtimestamp(float(start_point) - datetime_offset).strftime(time_str)[:-4]
        end_point_str = datetime.fromtimestamp(float(end_point) - datetime_offset).strftime(time_str)[:-4]
        
        if replace_time:
            old_start_point_str = string_out.split('\n')[-1].split(' - ')[0].split('[')[-1]
            word_sequence = string_out.split('\n')[-1].split(' - ')[-1].split(':')[-1].strip() + ' '
            string_out = '\n'.join(string_out.split('\n')[:-1])
            time_str = "[{} - {}]".format(old_start_point_str, end_point_str)
        else:
            time_str = "[{} - {}]".format(start_point_str, end_point_str)
            word_sequence = ''
        
        if not params['print_time']:
            time_str = ''
        
        strd = "\n{}{} {}: {}".format(color, time_str, speaker, word_sequence)
        return string_out + strd
    
    @staticmethod
    def print_word_colored(string_out, word, params, space=" "):
        word = word.strip()
        if params['print_transcript']:
            print(word, end=" ")
        return string_out + " " +  word
    
    def OnlineCOSclustering(
        self,
        uniq_embs_and_timestamps,
        oracle_num_speakers=None,
        max_num_speaker: int = 8,
        min_samples_for_NMESC: int = 6,
        enhanced_count_thres: int = 80,
        max_rp_threshold: float = 0.15,
        sparse_search_volume: int = 30,
        fixed_thres: float = 0.0,
        cuda=False,
    ):
        """
        Clustering method for speaker diarization based on cosine similarity.
        NME-SC part is converted to torch.tensor based operations in NeMo 1.9.

        Args:
            uniq_embs_and_timestamps: (dict)
                The dictionary containing embeddings, timestamps and multiscale weights.
                If uniq_embs_and_timestamps contains only one scale, single scale diarization
                is performed.

            oracle_num_speaker: (int or None)
                The oracle number of speakers if known else None

            max_num_speaker: (int)
                The maximum number of clusters to consider for each session

            min_samples_for_NMESC: (int)
                The minimum number of samples required for NME clustering. This avoids
                zero p_neighbour_lists. If the input has fewer segments than min_samples,
                it is directed to the enhanced speaker counting mode.

            enhanced_count_thres: (int)
                For the short audio recordings under 60 seconds, clustering algorithm cannot
                accumulate enough amount of speaker profile for each cluster.
                Thus, getEnhancedSpeakerCount() employs anchor embeddings (dummy representations)
                to mitigate the effect of cluster sparsity.
                enhanced_count_thres = 80 is recommended.

            max_rp_threshold: (float)
                Limits the range of parameter search.
                Clustering performance can vary depending on this range.
                Default is 0.15.

            sparse_search_volume: (int)
                Number of p_values we search during NME analysis.
                Default is 30. The lower the value, the faster NME-analysis becomes.
                Lower than 20 might cause a poor parameter estimation.

            fixed_thres: (float)
                If fixed_thres value is provided, NME-analysis process will be skipped.
                This value should be optimized on a development set to obtain a quality result.
                Default is None and performs NME-analysis to estimate the threshold.

        Returns:
            Y: (torch.tensor[int])
                Speaker label for each segment.
        """
        device = torch.device("cuda") if cuda else torch.device("cpu")

        # Get base-scale (the highest index) information from uniq_embs_and_timestamps.
        uniq_scale_dict = uniq_embs_and_timestamps['scale_dict']
        emb = uniq_scale_dict[max(uniq_scale_dict.keys())]['embeddings']
      
        emb = emb.to(emb.device)
        mat = getCosAffinityMatrix(emb)
        org_mat = copy.deepcopy(mat)
        
        emb, mat = emb.cpu().numpy(), mat.cpu().numpy()
        emb, reduced_labels, add_new = self.getReducedMat(mat, emb)
        emb = torch.tensor(emb).to(device)
        
        self.index_dict[self.frame_index] = (org_mat.shape[0], self.history_buffer_seg_end)
        self.index_dict['max_embed_count'] = max(org_mat.shape[0], self.index_dict['max_embed_count'])

        if emb.shape[0] == 1:
            return torch.zeros((1,), dtype=torch.int32)
        elif emb.shape[0] <= max(enhanced_count_thres, min_samples_for_NMESC) and oracle_num_speakers is None:
            est_num_of_spk_enhanced = getEnhancedSpeakerCount(emb, device)
        else:
            est_num_of_spk_enhanced = None

        if oracle_num_speakers:
            max_num_speaker = oracle_num_speakers

        mat, emb = getMultiScaleCosAffinityMatrix(uniq_embs_and_timestamps, device)
        print("Clustering embedding size:", emb.shape, mat.shape)
        nmesc = NMESC(
            mat,
            max_num_speaker=max_num_speaker,
            max_rp_threshold=max_rp_threshold,
            sparse_search=True,
            sparse_search_volume=sparse_search_volume,
            fixed_thres=fixed_thres,
            NME_mat_size=300,
            device=device,
        )

        if emb.shape[0] > min_samples_for_NMESC:
            est_num_of_spk, p_hat_value = nmesc.NMEanalysis()
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            affinity_mat = mat

        if oracle_num_speakers:
            est_num_of_spk = oracle_num_speakers
        elif est_num_of_spk_enhanced:
            est_num_of_spk = est_num_of_spk_enhanced

        spectral_model = SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda, device=device)
        Y = spectral_model.predict(affinity_mat)
        Y = Y.cpu().numpy()
        Y_out = self.matchLabels(org_mat, Y, add_new)
        return Y

    
    def _OnlineCOSclustering(self, key, emb, oracle_num_speakers=None, max_num_speaker=8, enhanced_count_thres=80, min_samples_for_NMESC=6, fixed_thres=None, cuda=False):
        """
        Online clustering method for speaker diarization based on cosine similarity.

        Parameters:
            key: (str)
                A unique ID for each speaker

            emb: (numpy array)
                Speaker embedding extracted from an embedding extractor

            oracle_num_speaker: (int or None)
                Oracle number of speakers if known else None

            max_num_speaker: (int)
                Maximum number of clusters to consider for each session

            min_samples: (int)
                Minimum number of samples required for NME clustering, this avoids
                zero p_neighbour_lists. Default of 6 is selected since (1/rp_threshold) >= 4
                when max_rp_threshold = 0.25. Thus, NME analysis is skipped for matrices
                smaller than (min_samples)x(min_samples).
        Returns:
            Y: (List[int])
                Speaker label for each segment.
        """
        mat = getCosAffinityMatrix(emb)
        org_mat = copy.deepcopy(mat)
        emb, reduced_labels, add_new = self.getReducedMat(mat, emb)
        
        self.index_dict[self.frame_index] = (org_mat.shape[0], self.history_buffer_seg_end)
        self.index_dict['max_embed_count'] = max(org_mat.shape[0], self.index_dict['max_embed_count'])

        if emb.shape[0] == 1:
            return np.array([0])
        elif emb.shape[0] <= max(enhanced_count_thres, min_samples_for_NMESC) and oracle_num_speakers is None:
            est_num_of_spk_enhanced = getEnhancedSpeakerCount(key, emb, cuda, random_test_count=5, anchor_spk_n=3, anchor_sample_n=10, sigma=100)
        else:
            est_num_of_spk_enhanced = None

        if oracle_num_speakers:
            max_num_speaker = oracle_num_speakers

        mat = getCosAffinityMatrix(emb)
        nmesc = NMESC(
            mat,
            max_num_speaker=max_num_speaker,
            max_rp_threshold=0.25,
            sparse_search=True,
            sparse_search_volume=10,
            fixed_thres=None,
            NME_mat_size=300,
            cuda=cuda,
        )

        if emb.shape[0] > min_samples_for_NMESC:
            est_num_of_spk, p_hat_value = self.estNumOfSpeakers(nmesc)
            affinity_mat = getAffinityGraphMat(mat, p_hat_value)
        else:
            est_num_of_spk, g_p = nmesc.getEigRatio(int(mat.shape[0]/2))
            affinity_mat = mat
        
        if oracle_num_speakers:
            est_num_of_spk = oracle_num_speakers
        elif est_num_of_spk_enhanced:
            est_num_of_spk = est_num_of_spk_enhanced

        spectral_model = _SpectralClustering(n_clusters=est_num_of_spk, cuda=cuda)
        Y = spectral_model.predict(affinity_mat)
        Y_out = self.matchLabels(org_mat, Y, add_new)
        return Y_out
    
    def estNumOfSpeakers(self, nmesc):
        """
        To save the running time, the p-value is only estimated in the beginning of the session.
        After switching to online mode, the system uses the most common estimated p-value.
        Args:
            nmesc: (NMESC)
                nmesc instance.
            isOnline: (bool)
                Indicates whether the system is running on online mode or not.

        Returns:
            est_num_of_spk: (int)
                The estimated number of speakers.
            p_hat_value: (int)
                The estimated p-value from NMESC method.
        """
        if self.isOnline:
            p_hat_value =  max(self.p_value_hist, key = self.p_value_hist.count)
            est_num_of_spk, g_p = nmesc.getEigRatio(p_hat_value)
        else:
            est_num_of_spk, p_hat_value, best_g_p_value = nmesc.NMEanalysis()
            self.p_value_hist.append(p_hat_value)
        return est_num_of_spk, p_hat_value
    
    def matchLabels(self, org_mat, Y, add_new):
        if self.isOnline:
            # Online clustering mode with history buffer
            update_point = self._history_buffer_segment_count
            Y_matched = self.matchNewOldclusterLabels(self.Y_fullhist[self.history_buffer_seg_end:], Y, with_history=True)
            if add_new:
                assert Y_matched[update_point:].shape[0] == self._current_buffer_segment_count, "Update point sync is not correct."
                Y_out = np.hstack((self.Y_fullhist[:self.history_buffer_seg_end], Y_matched[update_point:]))
                self.Y_fullhist = Y_out
            else:
                # Do not update cumulative labels since there are no new segments.
                Y_out = self.Y_fullhist[:org_mat.shape[0]]
            assert len(Y_out) == org_mat.shape[0], ipdb.set_trace()
        else:
            # If no memory is used, offline clustering is applied.
            Y_out = self.matchNewOldclusterLabels(self.Y_fullhist, Y, with_history=False)
            self.Y_fullhist = Y_out
        return Y_out

    @timeit
    def matchNewOldclusterLabels(self, Y_cumul, Y, with_history=True):
        """
        Run Hungarian algorithm (linear sum assignment) to find the best permuation mapping between
        the cumulated labels in history and the new clustering output labels.

        Args:
            Y_cumul (np.array):
                Cumulated diarization labels. This will be concatenated with history embedding speaker label
                then compared with the predicted label Y.

            Y (np.array):
                Contains predicted labels for reduced history embeddings concatenated with the predicted label.
                Permutation is not matched yet.

        Returns:
            mapping_array[Y] (np.array):
                An output numpy array where the input Y is mapped with mapping_array.

        """
        if len(Y_cumul) == 0:
            return Y
        spk_count = max(len(set(Y_cumul)), len(set(Y)))
        P_raw = np.hstack((self.history_embedding_buffer_label, Y_cumul)).astype(int)
        Q_raw = Y.astype(int)
        U_set = set(P_raw) | set(Q_raw)
        min_len = min(P_raw.shape[0], Q_raw.shape[0])
        P, Q = P_raw[:min_len], Q_raw[:min_len]
        PiQ, PuQ = (set(P) & set(Q)), (set(P) | set(Q))
        PmQ, QmP =  set(P) - set(Q),  set(Q) - set(P)
        
        # In len(PiQ) == 0 case, the label is totally flipped (0<->1)
        # without any commom labels.
        # This should be differentiated from the second case.
        if with_history and (len(PmQ) > 0 or len(QmP) > 0):
            # Keep only common speaker labels.
            # This is mainly for the newly added speakers from the labels in Y.
            keyQ = ~np.zeros_like(Q).astype(bool)
            keyP = ~np.zeros_like(P).astype(bool)
            for spk in list(QmP):
                keyQ[Q == spk] = False
            for spk in list(PmQ):
                keyP[P == spk] = False
            common_key = keyP*keyQ
            if all(~common_key) != True:
                P, Q = P[common_key], Q[common_key]
            elif all(~common_key) == True:
                P, Q = P, Q

        if len(U_set) == 1:
            # When two speaker vectors are exactly the same: No need to encode.
            mapping_array = np.array([0, 0])
            return mapping_array[Y]
        else:
            # Use one-hot encodding to find the best match.
            enc = OneHotEncoder(handle_unknown='ignore') 
            all_spks_labels = [[x] for x in range(len(U_set))]
            enc.fit(all_spks_labels)
            enc_P = enc.transform(P.reshape(-1, 1)).toarray()
            enc_Q = enc.transform(Q.reshape(-1, 1)).toarray()
            stacked = np.hstack((enc_P, enc_Q))
            cost = -1*linear_kernel(stacked.T)[spk_count:, :spk_count]
            row_ind, col_ind = linear_sum_assignment(cost)

            # If number of are speakers in each vector is not the same
            mapping_array = np.arange(len(U_set)).astype(int)
            for x in range(col_ind.shape[0]):
                if x in (set(PmQ) | set(QmP)):
                    mapping_array[x] = x
                else:
                    mapping_array[x] = col_ind[x]
            return mapping_array[Y]

class ASRWithDiarOnline:
    def __init__(self, 
                 diar, 
                 params, 
                 offset=10):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.params = params
        self.use_cuda = self.params['use_cuda']
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            self.cuda = True
        else: 
            self.device = torch.device("cpu")
            self.cuda = False
        self.sr = self.params['sample_rate']
        self.frame_len = float(self.params['frame_len'])
        self.frame_overlap = float(self.params['frame_overlap'])
        self.n_frame_len = int(self.frame_len * self.sr)
        self.n_frame_overlap = int(self.frame_overlap * self.sr)
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.buffer_length = self.buffer.shape[0]
        self.offset = offset
        self.CHUNK_SIZE = int(self.frame_len*self.sr)
        self.asr_batch_size = 16
        self._load_ASR_model(params)
        self._init_FrameBatchASR()
        self.ROUND = 2

        # For diarization
        self.diar = diar
        self.n_embed_seg_len = int(self.sr * self.diar.embed_seg_len)
        self.n_embed_seg_hop = int(self.sr * self.diar.embed_seg_hop)
         
        self.fine_embs_array = None
        self.frame_index = 0
        self.Y_fullhist = []
        
        # minimun width to consider non-speech activity 
        self.nonspeech_threshold = params['speech_detection_threshold']
        self.overlap_frames_count = int(self.n_frame_overlap/self.sr)
        self.cumulative_speech_labels = []

        self.buffer_start = None
        self.frame_start = 0
        self.rttm_file_path = None
        self.word_seq = []
        self.word_ts_seq = []
        self.merged_cluster_labels = []
        self.offline_logits = None
        self.debug_mode = False
        self.online_diar_label_update_sec = 30
        self.streaming_buffer_list = []
        self.reset()
        self.segment_ranges = []
        self.cluster_labels = []
    
    def _load_ASR_model(self, params):
        if 'citrinet' in  params['ASR_pretrained_model'].lower():
            self.asr_stride = 8
            self.asr_delay_sec = 0.12
            encdec_class = nemo_asr.models.EncDecCTCModelBPE
        elif 'conformer' in params['ASR_pretrained_model'].lower():
            self.asr_stride = 4
            self.asr_delay_sec = 0.06
            encdec_class = nemo_asr.models.EncDecCTCModelBPE
        else:
            raise ValueError(f"{params['ASR_pretrained_model']} is not compatible with the streaming launcher.")
        
        if '.nemo' in params['ASR_pretrained_model'].lower():
            self.new_asr_model = encdec_class.restore_from(restore_path=params['ASR_pretrained_model'], map_location=self.device)
        else:
            self.new_asr_model = encdec_class.from_pretrained(params['ASR_pretrained_model'], map_location=self.device)

        self.new_asr_model = self.new_asr_model.to(self.device)
        self.new_asr_model =self.new_asr_model.eval()
        self.time_stride = 0.01 * self.asr_stride
        self.params['offset'] = 0
        self.params['time_stride'] = self.asr_stride
        self.buffer_list = []

    def _load_LM_model(self):
        vocab_org = self.new_asr_model.decoder.vocabulary
        TOKEN_OFFSET = 100
        encoding_level = kenlm_utils.SUPPORTED_MODELS.get(type(self.new_asr_model).__name__, None)
        lm_path="/home/taejinp/Downloads/language_models/RIVA_A_confCTC_6gram_model.model"
        if encoding_level == "subword":
            vocab = [chr(idx + TOKEN_OFFSET) for idx in range(len(vocab_org))]
         
        beam_search_lm = nemo_asr.modules.BeamSearchDecoderWithLM(
            vocab=vocab,
            beam_width=200,
            alpha=1.0,
            beta=0.0,
            cutoff_prob=1.0,
            num_cpus=1,
            lm_path=lm_path,
            input_tensor=False,
        )
        return beam_search_lm

    def _convert_to_torch_var(self, audio_signal):
        audio_signal = torch.stack(audio_signal).float().to(self.new_asr_model.device)
        audio_signal_lens = torch.from_numpy(np.array([self.n_embed_seg_len for k in range(audio_signal.shape[0])])).to(self.new_asr_model.device)
        return audio_signal, audio_signal_lens

    def _process_cluster_labels(self, new_segment_ranges, new_cluster_labels):
        new_cluster_labels = new_cluster_labels.tolist() 
        if self.diar.history_buffer_seg_end:
            del self.segment_ranges[self.diar.history_buffer_seg_end:]
            del self.cluster_labels[self.diar.history_buffer_seg_end:]
            self.segment_ranges.extend(new_segment_ranges[self.diar.history_buffer_seg_end:])
            self.cluster_labels.extend(new_cluster_labels[self.diar.history_buffer_seg_end:])
            import ipdb; ipdb.set_trace()
        else:
            self.segment_ranges = copy.deepcopy(new_segment_ranges)
            self.cluster_labels = copy.deepcopy(new_cluster_labels)
    
        assert len(self.cluster_labels) == len(self.segment_ranges), ipdb.set_trace()
        lines = []
        for idx, label in enumerate(self.cluster_labels):
            tag = 'speaker_' + str(label)
            lines.append(f"{self.segment_ranges[idx][0]} {self.segment_ranges[idx][1]} {tag}")
        
        cont_lines = self.get_contiguous_stamps(lines)
        string_labels = merge_stamps(cont_lines)
        return string_labels, self.segment_ranges
    
    def get_contiguous_stamps(self, stamps):
        """
        Return contiguous time stamps
        """
        lines = copy.deepcopy(stamps)
        contiguous_stamps = []
        for i in range(len(lines) - 1):
            start, end, speaker = lines[i].split()
            next_start, next_end, next_speaker = lines[i + 1].split()
            if float(end) > float(next_start):
                boundary = str((float(next_start) + float(end)) / 2.0)
                lines[i + 1] = ' '.join([boundary, next_end, next_speaker])
                contiguous_stamps.append(start + " " + boundary + " " + speaker)
            else:
                contiguous_stamps.append(start + " " + end + " " + speaker)
        start, end, speaker = lines[-1].split()
        contiguous_stamps.append(start + " " + end + " " + speaker)
        return contiguous_stamps
    
    def _update_word_and_word_ts(self, words, word_timetamps):
        """
        Use the longest common sequence algorithm to update the newly decoded ASR results.
        """
        if self.word_ts_seq == []:
            self.word_seq.extend(words)
            self.word_ts_seq.extend(word_timetamps)
        else:
            old_seq_len = len(words)
            old_seq_offset = max(len(self.word_seq) - len(words), 0)
            lcs_idx, lcs_alignment = lcs_alg(self.word_seq[-1*(old_seq_len):], words)
            slice_idx = lcs_idx[1] + lcs_idx[-1] 
            align_mat = np.stack(lcs_alignment)
            new_hyp_idxm = np.where(align_mat.sum(axis=0))[0]
            if len(new_hyp_idxm) == 0:
                new_stt, old_end = 0, len(self.word_seq)
            else:
                new_stt = new_hyp_idxm[0] - 1
                old_seq_idxm = np.where(align_mat[:,new_stt+1] > 0)[0]
                old_end = old_seq_offset + old_seq_idxm[0] - 1

            del self.word_seq[old_end:]
            del self.word_ts_seq[old_end:]
            self.word_seq.extend(words[new_stt:])
            self.word_ts_seq.extend(word_timetamps[new_stt:])
    
    @torch.no_grad()
    def _run_embedding_extractor(self, audio_signal):
        torch_audio_signal, torch_audio_signal_lens = self._convert_to_torch_var(audio_signal)
        _, torch_embs = self.diar._speaker_model.forward(input_signal=torch_audio_signal, 
                                                         input_signal_length=torch_audio_signal_lens)
        return torch_embs
    
    @timeit
    def _get_speaker_embeddings(self, hop, embs_array, audio_signal, segment_ranges, online_extraction=True):
        if online_extraction:
            if embs_array is None:
                target_segment_count = len(segment_ranges)
                stt, end = 0, len(segment_ranges)
            else:
                target_segment_count = int(min(np.ceil((2*self.frame_overlap + self.frame_len)/hop), len(segment_ranges)))
                stt, end = len(segment_ranges)-target_segment_count, len(segment_ranges)
             
            if end > stt:
                torch_embs = self._run_embedding_extractor(audio_signal[stt:end])
                if embs_array is None:
                    embs_array = torch_embs
                else:
                    # embs_array = np.vstack((embs_array[:stt,:], torch_embs.cpu().numpy()))
                    embs_array = torch.vstack((embs_array[:stt,:], torch_embs))
            assert len(segment_ranges) == embs_array.shape[0], "Segment ranges and embs_array shapes do not match."
            
        else:
            torch_embs = self._run_embedding_extractor(audio_signal)
            embs_array = torch_embs.cpu().numpy()
        return embs_array

    @timeit
    def online_diarization(self, vad_ts):
        if self.buffer_start < 0:
        # or len(audio_signal) == 0:
            return []
        
        for scale_idx, (window, shift) in self.diar.multiscale_args_dict['scale_dict'].items():
            # audio_signal, segment_ranges = self.get_diar_segments(vad_ts, 
            # self.diar.segment_raw_audio_list[self.diar.uniq_id], self.diar.segment_abs_time_range_list[scale_idx] 
            audio_signal, segment_ranges = self.get_diar_segments(vad_ts, 
                                                                self.diar.segment_raw_audio_list[self.diar.uniq_id][scale_idx],
                                                                self.diar.segment_abs_time_range_list[self.diar.uniq_id][scale_idx], 
                                                                window, 
                                                                shift)
            # self.diar.embs_array[self.diar.uniq_id][scale_idx] 
            embeddings = self._get_speaker_embeddings(shift, 
                                                    self.diar.embs_array[self.diar.uniq_id][scale_idx], 
                                                    self.diar.segment_raw_audio_list[self.diar.uniq_id][scale_idx], 
                                                    self.diar.segment_abs_time_range_list[self.diar.uniq_id][scale_idx] )
            
            # if type(embeddings) != type(np.array([])):
            # embeddings = torch.tensor(embeddings).to(self.diar.device) 
            embeddings = embeddings.to(self.diar.device) 
            self.diar.embs_array[self.diar.uniq_id][scale_idx] = embeddings
            self.diar.segment_abs_time_range_list[self.diar.uniq_id][scale_idx] = segment_ranges
            self.diar.segment_raw_audio_list[self.diar.uniq_id][scale_idx] = audio_signal
            segment_ranges_str = [ f'{start:.3f} {end:.3f} ' for (start, end) in segment_ranges ]
            self.diar.multiscale_embeddings_and_timestamps[scale_idx] = [{self.diar.uniq_id: embeddings}, {self.diar.uniq_id: segment_ranges_str}]
        
        embs_and_timestamps = get_embs_and_timestamps(
            self.diar.multiscale_embeddings_and_timestamps, self.diar.multiscale_args_dict
        )
        
        # _diarization_function = COSclustering
        _diarization_function = self.diar.OnlineCOSclustering

        # cluster_labels = _diarization_function(
            # None, 
            # self.embs_array, 
            # oracle_num_speakers=self.diar.oracle_num_speakers,
            # enhanced_count_thres=self.diar.enhanced_count_thres, 
            # max_num_speaker=self.diar.max_num_speaker, 
            # cuda=self.cuda,
        # )
        
        cluster_labels = _diarization_function(
            embs_and_timestamps[self.diar.uniq_id], 
            oracle_num_speakers=self.diar.oracle_num_speakers,
            enhanced_count_thres=self.diar.enhanced_count_thres, 
            max_num_speaker=self.diar.max_num_speaker, 
            cuda=True,
        )
        assert len(cluster_labels) == self.diar.embs_array[self.diar.uniq_id][self.diar.base_scale_index].shape[0]

        # string_labels = self._process_cluster_labels(segment_ranges, cluster_labels)
        string_labels, segment_ranges = self._process_cluster_labels(self.diar.segment_abs_time_range_list[self.diar.uniq_id][self.diar.base_scale_index], cluster_labels)
        self.diar.segment_abs_time_range_list[self.diar.uniq_id][self.diar.base_scale_index] = segment_ranges
        return string_labels
    
    def _get_speaker_label_per_word(self, uniq_id, words, word_ts_list, pred_diar_labels):
        params = self.diar.params
        start_point, end_point, speaker = pred_diar_labels[0].split()
        word_pos, idx = 0, 0
        string_out = ''
        string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params)
        for j, word_ts_stt_end in enumerate(word_ts_list):
            word_pos = np.mean(word_ts_stt_end)
            if word_pos < float(end_point):
                string_out = self.diar.print_word_colored(string_out, words[j], params)
            else:
                idx += 1
                idx = min(idx, len(pred_diar_labels)-1)
                old_speaker = speaker
                start_point, end_point, speaker = pred_diar_labels[idx].split()
                if speaker != old_speaker:
                    string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params)
                else:
                    string_out = self.diar.print_time_colored(string_out, speaker, start_point, end_point, params, replace_time=True)
                string_out = self.diar.print_word_colored(string_out, words[j], params)

            stt_sec, end_sec = self.get_timestamp_in_sec(word_ts_stt_end, params)
        string_out = self.break_lines(string_out)
        string_out = string_out.replace('#', ' ')
        if self.rttm_file_path:
            string_out = self._print_DER_info(uniq_id, string_out, pred_diar_labels, params)
        else:
            logging.info(
                "Streaming Diar [{}][frame-  {}th  ]:".format(
                    self.diar.uniq_id, self.frame_index
                )
            )

        return string_out 
    
    @staticmethod
    def break_lines(string_out, max_line_N=90):
        split_string_out = string_out.split('\n')
        # last_line = split_string_out.pop()
        return_string_out = []
        for org_chunk in split_string_out:
            buffer = []
            color_str = org_chunk[:7]
            chunk = org_chunk[7:]
            if len(chunk) > max_line_N:
                for i in range(0, len(chunk), max_line_N):
                    buffer.append(color_str+chunk[i:i+max_line_N])
                return_string_out.extend(buffer)
            else:
                return_string_out.append(org_chunk)
        return '\n'.join(return_string_out)

    @staticmethod
    def get_timestamp_in_sec(word_ts_stt_end, params):
        stt = round(params['offset'] + word_ts_stt_end[0] * params['time_stride'], params['round_float'])
        end = round(params['offset'] + word_ts_stt_end[1] * params['time_stride'], params['round_float'])
        return stt, end
    
    
    def _print_DER_info(self, uniq_id, string_out, pred_diar_labels, params):
        if params['color']:
            color = self.diar.color_palette['white']
        else:
            color = ''
        der_dict, der_stat_dict = self.diar.online_eval_diarization(pred_diar_labels, self.rttm_file_path)
        DER, FA, MISS, CER = der_dict['DER'], der_dict['FA'], der_dict['MISS'], der_dict['CER']
        string_out += f'\n{color}============================================================================='
        string_out += f'\n{color}[Session: {uniq_id}, DER:{DER:.2f}%, FA:{FA:.2f}% MISS:{MISS:.2f}% CER:{CER:.2f}%]'
        string_out += f'\n{color}[Num of Speakers (Est/Ref): {der_stat_dict["est_n_spk"]}/{der_stat_dict["ref_n_spk"]}]'
        self.diar.DER_csv_list.append(f"{self.frame_index}, {DER}, {FA}, {MISS}, {CER}\n")
        write_txt(f"{asr_diar.diar._out_dir}/{uniq_id}.csv", ''.join(self.diar.DER_csv_list))
        return string_out
    
    def update_frame_to_buffer(self, frame): 
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        assert len(frame)==self.n_frame_len
        self.buffer_start = round(float(self.frame_index - 2*self.overlap_frames_count), 2)
        self.buffer[:-self.n_frame_len] = copy.deepcopy(self.buffer[self.n_frame_len:])
        self.buffer[-self.n_frame_len:] = copy.deepcopy(frame)
        
    @timeit
    def get_speech_labels_from_decoded_prediction(self, input_word_ts):
        speech_labels = []
        word_ts = copy.deepcopy(input_word_ts)
        if word_ts == []:
            return speech_labels
        else:
            count = len(word_ts)-1
            while count > 0:
                if len(word_ts) > 1: 
                    if word_ts[count][0] - word_ts[count-1][1] <= self.nonspeech_threshold:
                        trangeB = word_ts.pop(count)
                        trangeA = word_ts.pop(count-1)
                        word_ts.insert(count-1, [trangeA[0], trangeB[1]])
                count -= 1
        return word_ts 
    
    def set_buffered_infer_params(self, asr_model: Type[EncDecCTCModelBPE], frame_asr) -> Tuple[float, float, float]:
        """
        Prepare the parameters for the buffered inference.
        """
        cfg = copy.deepcopy(asr_model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)

        # some changes for streaming scenario
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"

        preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        preprocessor.to(asr_model.device)
        frame_asr.raw_preprocessor = preprocessor

        # Disable config overwriting
        OmegaConf.set_struct(cfg.preprocessor, True)
        self.offset_to_mid_window = (self.total_buffer_in_secs - self.chunk_len_in_sec) / 2
        self.onset_delay = (
            math.ceil(((self.total_buffer_in_secs - self.chunk_len_in_sec) / 2) / self.model_stride_in_secs) + 1
        )
        self.mid_delay = math.ceil(
            (self.chunk_len_in_sec + (self.total_buffer_in_secs - self.chunk_len_in_sec) / 2)
            / self.model_stride_in_secs
        )
        self.tokens_per_chunk = math.ceil(self.chunk_len_in_sec / self.model_stride_in_secs)
   
    def _init_FrameBatchASR(self):
        torch.manual_seed(0)
        torch.set_grad_enabled(False)

        # buffer_delay = self.overlap_frames_count
        self.chunk_len_in_sec = self.frame_len
        context_len_in_secs = self.frame_overlap
        self.total_buffer_in_secs = 2*context_len_in_secs + self.chunk_len_in_sec
        self.model_stride_in_secs = 0.04

        self.werbpe_ts = WERBPE_TS(
            tokenizer=self.new_asr_model.tokenizer,
            batch_dim_index=0,
            use_cer=self.new_asr_model._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self.new_asr_model._cfg.get("log_prediction", False),
        )
            
        self.frame_asr = FrameBatchASR_Logits_Sample(
            asr_model=self.new_asr_model,
            frame_len=self.chunk_len_in_sec,
            total_buffer=self.total_buffer_in_secs,
            batch_size=self.asr_batch_size,
        )
        self.frame_asr.reset()

        self.set_buffered_infer_params(self.new_asr_model, self.frame_asr)
        self.onset_delay_in_sec = round(self.onset_delay * self.model_stride_in_secs, 2)
    
    @timeit
    def _run_VAD_decoder(self, buffer):
        """
        Place holder for VAD integration. This function returns vad_mask that is identical for ASR feature matrix for
        the current buffer.
        """
        logging.info(f"Running VAD model {self.params['ASR_pretrained_model']}")
        hyps, tokens_list, feats_shape = get_wer_feat_logit_single(buffer,
                                                    self.frame_asr,
                                                    self.chunk_len_in_sec,
                                                    self.tokens_per_chunk,
                                                    self.mid_delay,
                                                    self.model_stride_in_secs,
                                                    frame_mask=None,
                                                )
        vad_mask = torch.ones(feats_shape)
        vad_timestamps = None
        return vad_mask, vad_timestamps

    @timeit
    def _run_ASR_decoder(self, buffer, frame_mask):
        self.decoder_offset = 0.0
        logging.info(f"Running ASR model {self.params['ASR_pretrained_model']}")
        hyps, tokens_list, _  = get_wer_feat_logit_single(buffer,
                                                    self.frame_asr,
                                                    self.chunk_len_in_sec,
                                                    self.tokens_per_chunk,
                                                    self.mid_delay,
                                                    self.model_stride_in_secs,
                                                    frame_mask,
                                                )
        greedy_predictions_list = tokens_list[0]
        logits_len = torch.from_numpy(np.array([len(greedy_predictions_list)]))
        greedy_predictions = torch.from_numpy(np.array(greedy_predictions_list)).unsqueeze(0)
        text, char_ts, _word_ts = self.werbpe_ts.ctc_decoder_predictions_tensor_with_ts(
            self.model_stride_in_secs, greedy_predictions, predictions_len=logits_len
        )
        words, word_ts = text[0].split(), _word_ts[0]
        assert len(words) == len(word_ts)
        self.offset_comp  = self.onset_delay_in_sec + (self.total_buffer_in_secs - self.chunk_len_in_sec)
        self.asr_offset = self.buffer_start - self.onset_delay_in_sec
        word_ts_adj = [[round(x[0] + self.asr_offset,2), round(x[1] + self.asr_offset,2)] for x in word_ts]
        return words, word_ts_adj

    @timeit
    def get_diar_segments(self, vad_timestamps, segment_raw_audio_list, segment_abs_time_range_list, window, shift):
        """
        Remove the old segments that overlap with the new frame (self.frame_start)
        cursor_for_old_segments is set to the onset of the t_range popped most recently.

        Frame is in the middle of the buffer.

        |___Buffer___[   Frame   ]___Buffer___|

        """
        if self.buffer_start >= 0:
            self._get_update_abs_time()
            
            if segment_raw_audio_list == [] and vad_timestamps != []:
                vad_timestamps[0][0] = max(vad_timestamps[0][0], 0.0)
                speech_labels_for_update = copy.deepcopy(vad_timestamps)
                self.cumulative_speech_labels = speech_labels_for_update
            
            else: 
                cursor_for_old_segments = self._get_new_cursor_for_update(segment_raw_audio_list, segment_abs_time_range_list)
                speech_labels_for_update = self._get_speech_labels_for_update(vad_timestamps,
                                                                              cursor_for_old_segments)
                
            source_buffer = copy.deepcopy(self.buffer)
            sigs_list, sig_rangel_list = self.get_segments_from_buffer(speech_labels_for_update, 
                                                                       source_buffer, 
                                                                       window, 
                                                                       shift)
            
            segment_raw_audio_list.extend(sigs_list)
            segment_abs_time_range_list.extend(sig_rangel_list)
                
        return segment_raw_audio_list, \
               segment_abs_time_range_list
        
            # self.diar.segment_raw_audio_list.extend(sigs_list)
            # self.diar.segment_abs_time_range_list.extend(sig_rangel_list)
                
        # return self.diar.segment_raw_audio_list, \
               # self.diar.segment_abs_time_range_list
    
    def _get_new_cursor_for_update(self, segment_raw_audio_list, segment_abs_time_range_list):
        """
        Remove the old segments that overlap with the new frame (self.frame_start)
        cursor_for_old_segments is set to the onset of the t_range popped lastly.
        """
        cursor_for_old_segments = self.frame_start
        while True and len(segment_raw_audio_list) > 0:
            t_range = segment_abs_time_range_list[-1]

            mid = np.mean(t_range)
            if self.frame_start <= t_range[1]:
                segment_abs_time_range_list.pop()
                segment_raw_audio_list.pop()
                cursor_for_old_segments = t_range[0]
            else:
                break
        return cursor_for_old_segments
        # cursor_for_old_segments = self.frame_start
        # while True and len(self.diar.segment_raw_audio_list) > 0:
            # t_range = self.diar.segment_abs_time_range_list[-1]

            # mid = np.mean(t_range)
            # if self.frame_start <= t_range[1]:
                # self.diar.segment_abs_time_range_list.pop()
                # self.diar.segment_raw_audio_list.pop()
                # cursor_for_old_segments = t_range[0]
            # else:
                # break
        # return cursor_for_old_segments

    def _get_update_abs_time(self):
        new_bufflen_sec = self.n_frame_len / self.sr
        n_buffer_samples = int(len(self.buffer)/self.sr)
        total_buffer_len_sec = n_buffer_samples/self.frame_len
        self.buffer_end = self.buffer_start + total_buffer_len_sec
        self.frame_start = round(self.buffer_start + int(self.n_frame_overlap/self.sr), self.ROUND)

    def _get_speech_labels_for_update(self, vad_timestamps, cursor_for_old_segments):
        """
        Bring the new speech labels from the current buffer. Then

        1. Concatenate the old speech labels from self.cumulative_speech_labels for the overlapped region.
            - This goes to new_speech_labels.
        2. Update the new 1 sec of speech label (speech_label_for_new_segments) to self.cumulative_speech_labels.
        3. Return the speech label from cursor_for_old_segments to buffer end.

        """
        if cursor_for_old_segments < self.frame_start:
            update_overlap_range = [cursor_for_old_segments, self.frame_start]
        else:
            update_overlap_range = []

        new_incoming_speech_labels = getSubRangeList(target_range=(self.frame_start, self.buffer_end),
                                                     source_list=vad_timestamps)

        update_overlap_speech_labels = getSubRangeList(target_range=update_overlap_range, 
                                                       source_list=self.cumulative_speech_labels)
        
        speech_label_for_new_segments = getMergedRanges(update_overlap_speech_labels, 
                                                             new_incoming_speech_labels) 
        
        self.cumulative_speech_labels = getMergedRanges(self.cumulative_speech_labels, 
                                                             new_incoming_speech_labels) 
        return speech_label_for_new_segments
    
    def get_num_of_seg_slices(self, n_dur_samples, n_seghop_samples):
        """
        Calculate the number of segments to be generated from the given VAD timestamps.
        """
        assert n_dur_samples > 0, f"n_dur_samples: {n_dur_samples} should be greater than zero."
        if n_dur_samples <= n_seghop_samples:
            slices = 1
        elif n_dur_samples % n_seghop_samples == 0:
            slices = n_dur_samples // n_seghop_samples
        else:
            slices = (n_dur_samples // n_seghop_samples) + 1
        return slices
    

    def get_segments_from_buffer(self, speech_labels_for_update, source_buffer, window, shift):
        sigs_list, sig_rangel_list = [], []

        # n_seglen_samples = int(self.diar.embed_seg_len*self.sr)
        # n_seghop_samples = int(self.diar.embed_seg_hop*self.sr)
        n_seglen_samples = int(window * self.sr)
        n_seghop_samples = int(shift * self.sr)
        src_len = source_buffer.shape[0] 
        for idx, range_t in enumerate(speech_labels_for_update):
            range_t = [range_t[0] - self.buffer_start, range_t[1] - self.buffer_start]
            range_t[0] = max(0, range_t[0])
            stt_b = min(int(range_t[0] * self.sr), src_len)
            end_b = min(int(range_t[1] * self.sr), src_len)
            if range_t[1] < 0 or stt_b == end_b:
                continue
            
            sigs, sig_lens = [], []
            n_dur_samples = int(end_b - stt_b)
            slices = self.get_num_of_seg_slices(n_dur_samples, n_seghop_samples)
            target_sig = torch.from_numpy(source_buffer[stt_b:end_b])
            sigs, sig_lens = self.get_segments_from_slices(slices, 
                                                           target_sig,
                                                           n_seglen_samples,
                                                           n_seghop_samples, 
                                                           sigs, 
                                                           sig_lens)
            assert len(sigs) == len(sig_lens)
            sigs_list.extend(sigs)
            segment_offset = self.buffer_start
            for seg_idx, sig_len in enumerate(sig_lens):
                seg_len_sec = float(sig_len / self.sr)
                start_abs_sec = round(float(segment_offset + seg_idx*self.diar.embed_seg_hop), self.ROUND)
                end_abs_sec = round(float(segment_offset + seg_idx*self.diar.embed_seg_hop + seg_len_sec), self.ROUND)
                sig_rangel_list.append([start_abs_sec, end_abs_sec])
        assert len(sigs_list) == len(sig_rangel_list)
        return sigs_list, sig_rangel_list

    def get_segments_from_slices(self, slices, sig, slice_length, shift, audio_signal, audio_lengths):
        """create short speech segments from sclices
        Args:
            slices (int): the number of slices to be created
            slice_length (int): the lenghth of each slice
            shift (int): the amount of slice window shift
            sig (FloatTensor): the tensor that contains input signal

        Returns:
            audio_signal (list): list of sliced input signal
            audio_lengths (list): list of audio sample lengths
        """
        last_seg_stt = shift * (slices-1)
        assert last_seg_stt < len(sig), f"the start of the last segment {last_seg_stt} is out of range of sig {len(sig)}."
        for slice_id in range(slices):
            start_idx = int(slice_id * shift)
            end_idx = int(start_idx + slice_length)
            signal = sig[start_idx:end_idx]
            if len(signal) < slice_length:
                signal = repeat_signal(signal, len(signal), slice_length)
            audio_signal.append(signal)
            audio_lengths.append(len(signal))
            
        return audio_signal, audio_lengths
    
    @torch.no_grad()
    def transcribe(self, frame=None, merge=True):
        self.update_frame_to_buffer(frame)

        vad_mask, vad_ts = self._run_VAD_decoder(self.buffer) 
        text, word_ts = self._run_ASR_decoder(self.buffer, frame_mask=vad_mask)

        if vad_ts is None:
            vad_ts = self.get_speech_labels_from_decoded_prediction(word_ts)

        
        # pred_diar_labels = self.online_diarization(audio_signal, audio_lengths)
        pred_diar_labels = self.online_diarization(vad_ts)

        self.frame_index += 1
        return text, word_ts, pred_diar_labels
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

def get_session_list(diar_init, args):
    if args.single_audio_file_path:
        uniq_id = get_uniqname_from_filepath(args.single_audio_file_path)
        diar_init.AUDIO_RTTM_MAP = {uniq_id: {'audio_path': args.single_audio_file_path,
                                               'rttm_path': args.single_rttm_file_path}}
    session_list = [ x[0] for x in  diar_init.AUDIO_RTTM_MAP.items() ]
    return diar_init, session_list


if __name__ == "__main__":
    SPK_EMBED_MODEL="/disk2/ejrvs/model_comparision/titanet-l.nemo"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_speaker_model", default=SPK_EMBED_MODEL, type=str, help="")
    parser.add_argument("--use_cuda", default=False, type=bool, help="")
    parser.add_argument("--single_audio_file_path", default=None, type=str, help="")
    parser.add_argument("--single_rttm_file_path", default=None, type=str, help="")
    parser.add_argument("--audiofile_list_path", default=None, type=str, help="")
    parser.add_argument("--reference_rttmfile_list_path", default=None, type=str, help="")
    parser.add_argument("--diarizer_out_dir", type=str, help="")
    parser.add_argument("--color", default=True, type=str, help="")
    parser.add_argument("--print_time", default=True, type=str, help="")
    parser.add_argument("--force_real_time", default=False, type=str, help="")
    parser.add_argument("--frame_len", default=1, type=int, help="")
    parser.add_argument("--frame_overlap", default=2, type=int, help="")
    parser.add_argument("--round_float", default=2, type=int, help="")
    parser.add_argument("--max_word_ts_length_in_sec", default=0.7, type=float, help="")
    parser.add_argument("--window_length_in_sec", default=1.5, type=float, help="")
    parser.add_argument("--shift_length_in_sec", default=0.75, type=float, help="")
    parser.add_argument("--print_transcript", default=False, type=bool, help="")
    parser.add_argument("--lenient_overlap_WDER", default=True, type=bool, help="")
    parser.add_argument("--speech_detection_threshold", default=100, type=int, help="")
    parser.add_argument("--ASR_pretrained_model", default='stt_en_conformer_ctc_large', type=str, help="")
    parser.add_argument("--checkpoint", default=None, type=str, help="")
    parser.add_argument("--sample_rate", default=16000, type=int, help="")
    parser.add_argument("--online_diar_buffer_segment_quantity", default=200, type=int, help="")
    parser.add_argument("--online_history_buffer_segment_quantity", default=100, type=int, help="")
    parser.add_argument("--enhanced_count_thres", default=0, type=int, help="")
    parser.add_argument("--max_num_speaker", default=8, type=int, help="")
    args = parser.parse_args()

    overrides = [
    f"diarizer.speaker_embeddings.model_path={args.pretrained_speaker_model}",
    f"diarizer.out_dir={args.diarizer_out_dir}",
    f"diarizer.speaker_embeddings.parameters.window_length_in_sec={args.window_length_in_sec}",
    f"diarizer.speaker_embeddings.parameters.shift_length_in_sec={args.shift_length_in_sec}",
    ]
    params = vars(args)
    # params['use_cuda'] = False
    params['use_cuda'] = True
    hydra.initialize(config_path="conf")
    cfg_diar = hydra.compose(config_name="/online_diarization_with_asr.yaml", overrides=overrides)
    # import ipdb; ipdb.set_trace()
    diar = OnlineClusteringDiarizer(cfg=cfg_diar, params=params)
    asr_diar = ASRWithDiarOnline(diar, params, offset=4)
    diar.device = asr_diar.device
    asr_diar.reset()
    
    samplerate, sdata = wavfile.read(args.single_audio_file_path)
    asr_diar.rttm_file_path = args.single_rttm_file_path
    for i in range(int(np.floor(sdata.shape[0]/asr_diar.n_frame_len))):
        callback_sim(asr_diar, diar.uniq_id, i, sdata, frame_count=None, time_info=None, status=None)
