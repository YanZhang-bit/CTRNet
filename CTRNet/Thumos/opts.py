import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.00004)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--train_epochs',
        type=int,
        default=15)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=6)
    parser.add_argument(
        '--step_size',
        type=int,
        default=5)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    parser.add_argument(
        '--n_gpu',
        type=int,
        default=2)
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=8)

    # output settings
    parser.add_argument('--subset', type=str, default='validation')
    parser.add_argument('--output', type=str, default="./output")
    parser.add_argument('--weight', type=float, default=1.0)
    # dataset settings
    #自己加的
    parser.add_argument('--checkpoint', type=str,default="GTAD_checkpoint_9.pth.tar")
    parser.add_argument(
        '--video_info',
        type=str,
        default="./data/thumos_annotations/")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./data/thumos_annotations/")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=256) # 100 for anet
    parser.add_argument(
        '--feature_path',
        type=str,
        #default="./data/thumos_feature/feature_anet_200")
        #default="./data/thumos_feature/I3D")
        default="/data/zy/datasets/gtad")

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048)
    parser.add_argument(
        '--dilate',
        type=int,
        default=1)
        #default=400)


    # anchors
    parser.add_argument('--max_duration', type=int, default=64)  # anet: 100 snippets
    parser.add_argument('--min_duration', type=int, default=0)  # anet: 100 snippets

    parser.add_argument(
        '--skip_videoframes',
        type=int,
        default=5,
        help='the number of video frames to skip in between each one. using 1 means that there is no skip.'
    )

    # ablation experiments
    parser.add_argument(
           '--goi_samp',
           type=int,
           default=0) # 0: sample all frame; 1: sample each output position
    parser.add_argument(
           '--goi_style',
           type=int,
           default=1)  # 0: no context, 1: last layer context, 2: all layer context

    # localization branch
    parser.add_argument(
        '--kern_2d',
        type=int,
        default=3)
    parser.add_argument(
        '--pad_2d',
        type=int,
        default=1)

    # NMS
    parser.add_argument(
        '--nms_thr',
        type=float,
        default=1.5)

    parser.add_argument(
        '--start_thr',
        type=float,
        default=0.054)
    parser.add_argument(
        '--tiou_thr',
        type=float,
        default=0.1)
    parser.add_argument(
        '--end_thr',
        type=float,
        default=0.195)
    parser.add_argument(
        '--action_thr',
        type=float,
        default=0.15)
    parser.add_argument(
        '--repr',
        action='store_true',
        default=False,
        help="whether to use RePr training scheme")
    parser.add_argument('--S1',
                        type=int,
                        default=3,
                        help="S1 epochs for RePr")
    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)
    parser.add_argument('--multiscale', default=[1, 8, 16, 32], nargs='+', type=int)
    parser.add_argument('--S2',
                        type=int,
                        default=2,
                        help="S2 epochs for RePr")
    parser.add_argument('--prune_ratio',
                        type=float,
                        default=0.3,
                        help="prune ratio")
    parser.add_argument('--zero_init', action='store_true',
                        default=False,
                        help="whether to initialize with zero")
    # Override
    # In principle it should override the cache. However, the logic is to
    # always save a cache. Thus, the flag only prevents loading the cache 😂
    parser.add_argument(
        '--override', default=False, action='store_true',
        help='Prevent use of cached data'
    )

    args = parser.parse_args()

    return args

