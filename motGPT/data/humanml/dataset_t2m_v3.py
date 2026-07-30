import rich
import random
import pickle
import os
import numpy as np
import codecs as cs
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
import time

class Text2MotionDatasetCBV3(data.Dataset):
    def __init__(
        self,
        data_root, 
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        code_path='VQVAE',
        task_path=None,
        std_text=False,
        instruction_type='all',
        **kwargs,
    ):
        self.tiny = tiny
        self.unit_length = unit_length

        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        # Data mean and std
        self.mean = mean
        self.std = std
        self.fps = fps
        self.data_root = data_root

        # Data path - use the split parameter passed in, don't override it
        split_file = pjoin(data_root, split + '.txt')
        code_dir = pjoin(data_root, code_path)
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')
        self.motion_dir = motion_dir
        print(code_dir, motion_dir)
        
        instruction_type = '' if (instruction_type == 'all') else '_'+instruction_type
        if task_path:
            instructions = task_path
        elif stage in ['lm_pretrain','lm_adaptor_pretrain', "lm_t2m"]:
            instructions = pjoin(data_root, f'template{instruction_type}_pretrain.json')
        elif stage in ['lm_instruct', "lm_rl", "lm_finetune"]:
            instructions = pjoin(data_root, f'template{instruction_type}_instructions.json')
        else:
            raise NotImplementedError(f"stage {stage} not implemented")

        # Try to load pre-built lazy index for fast startup
        # (skipped in tiny/debug mode which only loads a small subset)
        self.lazy_load = False
        lazy_index_file = pjoin(data_root, 'lazy_index', f'{split}.pkl')

        if not (tiny or debug) and os.path.isfile(lazy_index_file):
            if self._try_load_lazy_index(lazy_index_file):
                self.lazy_load = True

        if not self.lazy_load:
            # Full file scan (first run or tiny/debug mode); saves lazy index for next time
            self._build_data_dict(data_root, split, split_file, motion_dir, text_dir,
                                  fps, tiny, debug, tmpFile, lazy_index_file)

        self.nlp = spacy.load('en_core_web_sm')
        self.std_text = std_text
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])
        
        print(f'dataset {split} loaded, {len(self.data_dict)}' + (' [lazy]' if self.lazy_load else ''))

    def _try_load_lazy_index(self, index_file):
        """Load pre-built lazy index (text + metadata only, no motion arrays)."""
        try:
            t0 = time.time()
            with open(index_file, 'rb') as f:
                index_data = pickle.load(f)
            params = index_data.get('params', {})
            if (params.get('min_motion_length') != self.min_motion_length or
                params.get('max_motion_length') != self.max_motion_length or
                params.get('fps') != self.fps):
                print(f"[LazyLoad] Index params mismatch, will rebuild.")
                return False
            self.name_list = index_data['name_list']
            self.data_dict = index_data['data_dict']
            print(f"[LazyLoad] Loaded {len(self.data_dict)} samples in {time.time() - t0:.1f}s")
            return True
        except Exception as e:
            print(f"[LazyLoad] Failed to load index: {e}. Falling back to full scan.")
            return False

    def _build_data_dict(self, data_root, split, split_file, motion_dir, text_dir,
                         fps, tiny, debug, tmpFile, lazy_index_file):
        """Scan all motion/text files to build data_dict. Saves lazy index for fast future loading."""
        # Data id list
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # Debug mode
        if tiny or debug:
            enumerator = enumerate(id_list)
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split}",
                ))
            maxdata = 1e10
            subset = ''

        new_name_list = []
        data_dict = {}
        lazy_data_dict = {}

        for i, name in enumerator:
            if len(new_name_list) > maxdata:
                break
            motion = np.load(pjoin(motion_dir, f'{name}.npy'))

            text_data = []
            flag = False
            with cs.open(pjoin(text_dir, name + '.txt')) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    t_tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict['caption'] = caption
                    text_dict['tokens'] = t_tokens
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        if int(f_tag * fps ) >= int(to_tag * fps): continue
                        motion_new = motion[int(f_tag * fps):int(to_tag * fps )]
                        new_name = '%s_%f_%f' % (name, f_tag, to_tag)

                        if (len(motion_new)) < self.min_motion_length or (
                                len(motion_new) >= self.max_motion_length):
                            continue
                        data_dict[new_name] = {
                            'motion': motion_new,
                            'text': [text_dict]
                        }
                        lazy_data_dict[new_name] = {
                            'text': [text_dict],
                            'base_name': name,
                            'motion_slice': (int(f_tag * fps), int(to_tag * fps)),
                        }
                        new_name_list.append(new_name)

            if flag and (not (len(motion)) < self.min_motion_length or (
                len(motion) >= self.max_motion_length)):
                data_dict[name] = {
                    'motion': motion,
                    'text': text_data
                }
                lazy_data_dict[name] = {
                    'text': text_data,
                    'base_name': name,
                    'motion_slice': None,
                }
                new_name_list.append(name)

        if tmpFile:
            try:
                os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
                with open(
                        pjoin(data_root,
                                f'tmp/{split}{subset}_data.pkl'),
                        'wb') as file:
                    pickle.dump(data_dict, file)
                with open(
                        pjoin(data_root,
                                f'tmp/{split}{subset}_index.pkl'),
                        'wb') as file:
                    pickle.dump(new_name_list, file)
            except OSError as e:
                print(f"Warning: failed to write cache files: {e}. Skipping cache.")

        # Save lazy index for fast subsequent runs (text + metadata only, no motion arrays)
        if not (tiny or debug):
            try:
                os.makedirs(os.path.dirname(lazy_index_file), exist_ok=True)
                index_data = {
                    'name_list': new_name_list,
                    'data_dict': lazy_data_dict,
                    'params': {
                        'min_motion_length': self.min_motion_length,
                        'max_motion_length': self.max_motion_length,
                        'fps': fps,
                    }
                }
                with open(lazy_index_file, 'wb') as f:
                    pickle.dump(index_data, f)
                print(f"[LazyLoad] Saved index to {lazy_index_file} ({len(lazy_data_dict)} samples)")
            except OSError as e:
                print(f"[LazyLoad] Warning: failed to save index: {e}")

        self.data_dict = data_dict
        self.name_list = new_name_list

    def _load_motion(self, fname):
        """Load motion for a sample. Returns pre-loaded array (eager) or loads from disk (lazy)."""
        entry = self.data_dict[fname]
        if 'motion' in entry:
            return entry['motion']
        motion = np.load(pjoin(self.motion_dir, f"{entry['base_name']}.npy"))
        if entry.get('motion_slice') is not None:
            start, end = entry['motion_slice']
            motion = motion[start:end]
        return motion

    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, item):
        # data_idx = item % len(self.name_list)
        # task_idx = item // len(self.name_list)
        data_idx = item // len(self.tasks)
        task_idx = item % len(self.tasks)

        fname = self.name_list[data_idx]
        data = self.data_dict[fname]
        text_list = data['text']
        motion = self._load_motion(fname)

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']
        if self.std_text:
            doc = self.nlp(caption)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN'
                        or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
                
            caption = ' '.join(word_list)
        
        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        # Random crop
        coin = np.random.choice([False, False, True])
        m_length = motion.shape[0]
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_length = (m_length // self.unit_length - 1) * self.unit_length
            else:
                m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std

        tasks = self.tasks[task_idx]

        return caption, None, None, motion, m_length, None, None, None, None, all_captions, tasks, fname
        # text, m_tokens, m_tokens_len, motion, length, word_embs, pos_ohot, text_len, tokens, all_captions ,tasks, fname
