import json
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(__file__, "../../mdm")))

import mdm.data_loaders.humanml.utils.paramUtil as paramUtil
import numpy as np
import requests  # type: ignore
import torch
from base.exception import MApplicationException
from base.logger import MLogger
from mdm.data_loaders.get_data import get_dataset_loader
from mdm.data_loaders.humanml.scripts.motion_process import recover_from_ric
from mdm.data_loaders.humanml.utils.plot_script import plot_3d_motion
from mdm.data_loaders.tensors import collate
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mdm.utils import dist_util
from mdm.utils.fixseed import fixseed
from mdm.utils.model_util import create_model_and_diffusion, load_model_wo_clip
from mdm.utils.parser_util import (add_base_options, add_data_options,
                                   add_diffusion_options, add_model_options,
                                   get_args_per_group_name,
                                   get_model_path_from_args)

from parts.config import DirName

logger = MLogger(__name__)


def execute(args):
    logger.info("text2motion開始", decoration=MLogger.DECORATION_BOX)

    if not args.text:
        logger.error(
            "指定された指示文が空です。\n{text}",
            text=args.text,
            decoration=MLogger.DECORATION_BOX,
        )
        raise MApplicationException()

    try:
        argv = sample_args()

        argv.output_dir = os.path.join(args.parent_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(argv.output_dir, exist_ok=True)

        argv.text_prompt = translate(args.text, args.lang)

        with open(os.path.join(argv.output_dir, "prompt.txt"), "w") as f:
            f.write(args.text)
            f.write("\n\n")
            f.write(translate(args.text, args.lang))
            f.write("\n\n")

        fixseed(argv.seed)
        out_path = argv.output_dir
        max_frames = 196 if argv.dataset in ['kit', 'humanml'] else 60
        fps = 30
        n_frames = min(max_frames, int(argv.motion_length * fps))
        dist_util.setup_dist(argv.device)

        print("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(argv)

        print(f"Loading checkpoints from [{argv.model_path}]...")
        state_dict = torch.load(argv.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

        if argv.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        model.to(dist_util.dev())
        model.eval()  # disable random masking

        print('Loading dataset...')
        texts = []
        if argv.text_prompt != '':
            texts = [argv.text_prompt]
            argv.num_samples = 1
        elif argv.input_text != '':
            assert os.path.exists(argv.input_text)
            with open(argv.input_text, 'r') as fr:
                texts = fr.readlines()
            texts = [s.replace('\n', '') for s in texts]
            argv.num_samples = len(texts)

        assert argv.num_samples <= argv.batch_size, \
            f'Please either increase batch_size({argv.batch_size}) or reduce num_samples({argv.num_samples})'
        # So why do we need this check? In order to protect GPU from a memory overload in the following line.
        # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
        # If it doesn't, and you still want to sample more prompts, run this script with different seeds
        # (specify through the --seed flag)
        argv.batch_size = argv.num_samples  # Sampling a single batch from the testset, with exactly argv.num_samples
        data = get_dataset_loader(name=argv.dataset,
                                    batch_size=argv.batch_size,
                                    num_frames=max_frames,
                                    split='test',
                                    hml_mode='text_only')
        data.fixed_length = n_frames
        total_num_samples = argv.num_samples * argv.num_repetitions

        _, model_kwargv = collate(
            [{'inp': torch.tensor([[0.]]), 'target': 0, 'text': txt, 'tokens': None, 'lengths': n_frames}
                for txt in texts]
        )

        all_motions = []
        all_lengths = []
        all_text = []

        for rep_i in range(argv.num_repetitions):
            print(f'### Start sampling [repetitions #{rep_i}]')

            # add CFG scale to batch
            if argv.guidance_param != 1:
                model_kwargv['y']['scale'] = torch.ones(argv.batch_size, device=dist_util.dev()) * argv.guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                (argv.batch_size, model.njoints, model.nfeats, n_frames),
                clip_denoised=False,
                model_kwargv=model_kwargv,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            all_text += model_kwargv['y']['text']
            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargv['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * argv.batch_size} samples")


        all_motions = np.concatenate(all_motions, axis=0)
        all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
        all_text = all_text[:total_num_samples]
        all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

        npy_path = os.path.join(out_path, 'results.npy')
        print(f"saving results file to [{npy_path}]")
        np.save(npy_path,
                {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
                    'num_samples': argv.num_samples, 'num_repetitions': argv.num_repetitions})
        with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
            fw.write('\n'.join(all_text))
        with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
            fw.write('\n'.join([str(l) for l in all_lengths]))

        print(f"saving visualizations to [{out_path}]...")
        skeleton = paramUtil.kit_kinematic_chain if argv.dataset == 'kit' else paramUtil.t2m_kinematic_chain
        for sample_i in range(argv.num_samples):
            for rep_i in range(argv.num_repetitions):
                caption = all_text[rep_i*argv.batch_size + sample_i]
                length = all_lengths[rep_i*argv.batch_size + sample_i]
                motion = all_motions[rep_i*argv.batch_size + sample_i].transpose(2, 0, 1)[:length]
                save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
                animation_save_path = os.path.join(out_path, save_file)
                print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
                if argv.dataset == 'kit':
                    motion *= 0.003  # scale for visualization
                elif argv.dataset == 'humanml':
                    motion *= 1.3  # scale for visualization
                plot_3d_motion(animation_save_path, skeleton, motion, title=caption, fps=fps)
                # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        abs_path = os.path.abspath(out_path)
        print(f'[Done] Results are at [{abs_path}]')

        logger.info(
            "text2motion完了: {text}",
            text=args.text,
            decoration=MLogger.DECORATION_BOX,
        )

        return True, argv.output_dir
    except Exception as e:
        logger.critical("text2motionで予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        raise e

def translate(text: str, lang: str):
    if lang == "en":
        return text

    params = (
        ("text", text),
        ("source", lang),
        ("target", "en"),
    )

    # GASを叩く
    # https://qiita.com/satto_sann/items/be4177360a0bc3691fdf
    response = requests.get(
        "https://script.google.com/macros/s/AKfycbzZtvOvf14TaMdRIYzocRcf3mktzGgXvlFvyczo/exec",
        params=params,
    )

    # 結果を解析
    results = json.loads(response.text)

    if "text" in results:
        return results["text"]

    return text


def sample_args():
    parser = ArgumentParser([])
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)

    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", default='../data/motion-diffusion-model/humanml_trans_enc_512/model000200000.pt', type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to csv/txt file that specifies generation. If empty, will take text prompts from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    argv = parser.parse_args(args=[])
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, argv, group_name)

    # load args from model
    args_path = os.path.join(os.path.abspath(os.path.dirname(argv.model_path)), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)
    for a in args_to_overwrite:
        if a in model_args.keys():
            argv.__dict__[a] = model_args[a]
        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, argv.__dict__[a]))

    return argv
