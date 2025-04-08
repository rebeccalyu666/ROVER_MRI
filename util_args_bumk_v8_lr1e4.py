import argparse
def get_args(cmd=True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description="NODF Estimation."
    )

    parser.add_argument(
        "--inr",
        action="store",
        type=str,
        help="Type of the INR - wire or siren or relu",
        choices=["siren", "wire", "relu"],
        default="siren",
    )

    parser.add_argument(
        "--device",
        action="store",
        type=str,
        help="Device.",
        default="cuda",
    )

    parser.add_argument(
        "--sh_order",
        action="store",
        default=8,
        type=int,
        help="Order of spherical harmonic basis",
    )

    parser.add_argument(
        "--bmarg",
        action="store",
        default=20,
        type=int,
        help="+= bmarg considered same b-value.",
    )

    parser.add_argument(
        "--rho",
        action="store",
        default=0.5,
        type=float,
        help="Length-scale parameter for Matern Prior.",
    )

    parser.add_argument(
        "--nu",
        action="store",
        default=1.5,
        type=float,
        help="Smoothness parameter for Matern Prior.",
    )

    parser.add_argument(
        "--num_epochs",
        action="store",
        default=20000,
        type=int,
        help="Number of trainging epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        action="store",
        default=0.0001,
        type=float,
        help="Learning rate for optimizer.",
    )

    parser.add_argument(
        "--calib_prop",
        action="store",
        default=0.1,
        type=float,
        help="Proportion of voxels to be used in posterior calibration.",
    )

    parser.add_argument(
        "--sigma2_mu",
        help="Variance for isotropic harmonic.",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--sigma2_w", help="Variance parameter for GP prior.", type=float, default=0.5
    )

    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--deconvolve", action="store_true")

    parser.add_argument("--enable_schedulers", action="store_true")

    parser.add_argument("--simulation", action="store_true")

    parser.add_argument(
        "--omega0", help="WIRE Gaussian parameter.", type=float, default=30.0
    )

    parser.add_argument(
        "--omega0_hidden",
        help="WIRE Gaussian parameter for hidden layers.",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--sigma0", help="WIRE Sine frequency parameter.", type=float, default=5.0
    )

    parser.add_argument("--skip_conn", action="store_true")

    parser.add_argument("--batchnorm", action="store_true")

    parser.add_argument(
        "--num_workers",
        action="store",
        default=12,
        type=int,
        help="Number of workers for the dataloader.",
    )

    parser.add_argument(
        "--per_level_scale",
        action="store",
        default=1.5,
        type=int,
        help="Per level scal of resolution.",
    )

    parser.add_argument("--weight_decay", help="Weight decay.", type=float, default=0.0)

    parser.add_argument(
        "--view_num",
        default=8,
        type=int,
        help="number of view.",
    )

    parser.add_argument(
        "--img_size",
        default=[480, 576, 98],
        help="size of matrix.",
    )

    parser.add_argument(
        "--img_path",
        action="store",
        default="../DWI_SR/BU_monkey_V8_1002/imgs_nii*",
        type=str,
        help="Nifti file for image (nx X ny X nz)",
    )

    parser.add_argument(
        "--batch_size",
        action="store",
        default=100000,
        type=int,
        help="batch_size.",
    )

    parser.add_argument(
        "--image_save_iter",
        action="store",
        default=1000,
        type=int,
        help="image_save_iter.",
    )
    parser.add_argument('--iter', type=int, default=15000, help="load model weights from iter")
    args = parser.parse_args()
    return args