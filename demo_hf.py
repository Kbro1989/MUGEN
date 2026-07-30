"""Run the released MUGEN model straight from HuggingFace.

No dataset, no checkpoint, and nothing from this repository is required: the
published model carries its own tokenizer and its own HumanML3D feature
statistics.

    # text -> motion, saved as joint positions
    python demo_hf.py --text "a person walks forward and then waves." --length 120

    # motion -> text, on a HumanML3D clip
    python demo_hf.py --motion_file datasets/humanml3d/new_joint_vecs/000004.npy

    # both directions on one clip (generate, then caption what came out)
    python demo_hf.py --text "a person jumps." --caption_output
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_REPO = "zy22b/MUGEN"


def main():
    parser = argparse.ArgumentParser(description="MUGEN HuggingFace demo")
    parser.add_argument("--model", default=DEFAULT_REPO,
                        help="HuggingFace repo id or a local directory")
    parser.add_argument("--text", nargs="*", default=None,
                        help="one or more motion descriptions to generate from")
    parser.add_argument("--length", type=int, default=120,
                        help="frames to generate at 20 fps (multiples of 4, up to 196)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature; default is the model's calibrated 1.0")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for a reproducible draw")
    parser.add_argument("--motion_file", default=None,
                        help="HumanML3D .npy clip (T, 263) to caption")
    parser.add_argument("--caption_output", action="store_true",
                        help="also caption the motion that was just generated")
    parser.add_argument("--out_dir", default="results/demo_hf",
                        help="where to write the generated joints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.text is None and args.motion_file is None:
        args.text = ["a person walks forward and then waves with the right hand."]

    print(f"Loading {args.model} on {args.device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"K = {model.config.k_latent_slots} latent slots")

    # ---------------------------------------------------------- text -> motion
    if args.text:
        generator = None
        if args.seed is not None:
            generator = torch.Generator(device=args.device).manual_seed(args.seed)

        features = model.generate_motion(
            args.text,
            lengths=args.length,
            tokenizer=tokenizer,
            temperature=args.temperature,
            generator=generator,
        )
        joints = model.features_to_joints(features)
        os.makedirs(args.out_dir, exist_ok=True)

        print(f"\nGenerated {tuple(features.shape)} features "
              f"-> {tuple(joints.shape)} joints")
        for i, prompt in enumerate(args.text):
            path = os.path.join(args.out_dir, f"sample_{i:02d}.npy")
            np.save(path, joints[i].cpu().numpy())
            root = joints[i, :, 0]
            travel = float((root[-1, [0, 2]] - root[0, [0, 2]]).norm())
            print(f"  [{i}] \"{prompt}\"")
            print(f"       root travel {travel:.2f} m -> {path}")

        if args.caption_output:
            captions = model.generate_caption(features, tokenizer=tokenizer)
            print("\nCaptioning the generated motion:")
            for prompt, caption in zip(args.text, captions):
                print(f"  prompt  : {prompt}")
                print(f"  caption : {caption}")

    # ---------------------------------------------------------- motion -> text
    if args.motion_file:
        motion = np.load(args.motion_file)
        clip = torch.from_numpy(motion).float().unsqueeze(0)
        caption = model.generate_caption(clip, tokenizer=tokenizer)[0]
        print(f"\n{args.motion_file}  {tuple(motion.shape)}")
        print(f"  caption : {caption}")

        reference = args.motion_file.replace("new_joint_vecs", "texts").replace(".npy", ".txt")
        if os.path.isfile(reference):
            with open(reference) as handle:
                first = handle.readline().split("#")[0].strip()
            print(f"  reference : {first}")


if __name__ == "__main__":
    main()
