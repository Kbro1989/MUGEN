#!/usr/bin/env python
"""
Pre-build dataset lazy index for fast training startup.

Usage:
    python data_preprocessing/build_dataset_index.py \
        --data_root datasets/humanml3d/ \
        --splits train val test

This scans all motion/text files once and saves a lightweight index
(text + metadata, no motion arrays) to {data_root}/lazy_index/{split}.pkl.

Subsequent training runs will load this index in ~1s instead of
scanning all files (~30min on slow disk).
"""

import argparse
import os
import pickle
import time
import numpy as np
import codecs as cs
from os.path import join as pjoin


def build_index(data_root, split, min_motion_length=20, max_motion_length=200, fps=20):
    """Build lazy index for a given split."""
    split_file = pjoin(data_root, split + '.txt')
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    text_dir = pjoin(data_root, 'texts')

    # Read id list
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())

    print(f"[{split}] Scanning {len(id_list)} samples...")
    t0 = time.time()

    name_list = []
    data_dict = {}

    for i, name in enumerate(id_list):
        if (i + 1) % 1000 == 0:
            print(f"  [{split}] {i+1}/{len(id_list)}...")

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
                    if int(f_tag * fps) >= int(to_tag * fps):
                        continue
                    motion_new = motion[int(f_tag * fps):int(to_tag * fps)]
                    new_name = '%s_%f_%f' % (name, f_tag, to_tag)

                    if len(motion_new) < min_motion_length or len(motion_new) >= max_motion_length:
                        continue
                    data_dict[new_name] = {
                        'text': [text_dict],
                        'base_name': name,
                        'motion_slice': (int(f_tag * fps), int(to_tag * fps)),
                    }
                    name_list.append(new_name)

        if flag and not (len(motion) < min_motion_length or len(motion) >= max_motion_length):
            data_dict[name] = {
                'text': text_data,
                'base_name': name,
                'motion_slice': None,
            }
            name_list.append(name)

    # Save index
    index_dir = pjoin(data_root, 'lazy_index')
    os.makedirs(index_dir, exist_ok=True)
    index_file = pjoin(index_dir, f'{split}.pkl')

    index_data = {
        'name_list': name_list,
        'data_dict': data_dict,
        'params': {
            'min_motion_length': min_motion_length,
            'max_motion_length': max_motion_length,
            'fps': fps,
        }
    }
    with open(index_file, 'wb') as f:
        pickle.dump(index_data, f)

    elapsed = time.time() - t0
    print(f"[{split}] Done: {len(data_dict)} samples indexed in {elapsed:.1f}s -> {index_file}")


def main():
    parser = argparse.ArgumentParser(description="Pre-build dataset lazy index for fast training startup")
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of HumanML3D dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help='Splits to index (default: train val test)')
    parser.add_argument('--min_motion_length', type=int, default=20)
    parser.add_argument('--max_motion_length', type=int, default=200)
    parser.add_argument('--fps', type=int, default=20)
    args = parser.parse_args()

    for split in args.splits:
        build_index(args.data_root, split,
                    min_motion_length=args.min_motion_length,
                    max_motion_length=args.max_motion_length,
                    fps=args.fps)

    print("\nAll done! Training will now start in seconds instead of minutes.")


if __name__ == '__main__':
    main()
