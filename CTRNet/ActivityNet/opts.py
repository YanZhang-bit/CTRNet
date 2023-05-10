import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./model')

    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.0001)
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
        default=12)
    parser.add_argument(
        '--step_size',
        type=int,
        default=7)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="data/activitynet_annotations/video_info_new.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="data/activitynet_annotations/anet_anno_action.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--feature_path',
        type=str,
        default="/data/zy/datasets/anet_1.3/")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="BMN_checkpoint_14.pth.tar")

    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--goi_samp',
        type=int,
        default=0)  # 0: sample all frame; 1: sample each output position
    parser.add_argument(
        '--goi_style',
        type=int,
        default=1)  # 0: no context, 1: last layer context, 2: all layer context
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)
    parser.add_argument('--max_duration', type=int, default=100)  # anet: 100 snippets

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=512)
    parser.add_argument(
        '--kern_2d',
        type=int,
        default=3)
    parser.add_argument(
        '--pad_2d',
        type=int,
        default=1)

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=8)
    parser.add_argument(
        '--tiou_thr',
        type=float,
        default=0.1)
    parser.add_argument(
        '--start_thr',
        type=float,
        default=0.054)
    parser.add_argument(
        '--end_thr',
        type=float,
        default=0.195)
    parser.add_argument(
        '--action_thr',
        type=float,
        default=0.15)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.28)
    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.59)
    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.93)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/result_proposal.json")
    parser.add_argument(
        '--result_file_pac',
        type=str,
        default="./output/result_proposal_pac.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="./output/evaluation_result.jpg")

    args = parser.parse_args()

    return args

