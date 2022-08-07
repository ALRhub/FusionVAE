import os
import yaml
from addict import Dict

cfg = Dict()

cfg.ds.shuffle_valid = False
cfg.misc.lr_schedule = True
cfg.misc.n_workers_train = 1
cfg.misc.n_workers_valid = 1


cfg.save.model = True
cfg.save.train_tb = 100  # save every n-th
cfg.save.rec_img = 1000
cfg.save.gen_imgs = False
cfg.save.max_n_rec_imgs = 100  # maximum number of reconstruction images to be saved e.g. 100 = 10x10
# save noisy input images together with target and reconstructed images:
cfg.save.grid.target_input_too = True  # input target input image together with noisy images
cfg.save.grid.target_input_alone = True   # input target input image only
cfg.save.grid.n_rows_per_category = 3 if cfg.args.dataset == 'mnist' else 2
cfg.save.grid.n_rows_per_category_final = 3
cfg.save.grid.n_img_per_temp_list = [1, 2, 2]  # number of images with temperature=[0, 0.7, 1] # number of reconstruction images (to be saved together with their noisy inputs, the target image, and a mean image)
cfg.save.grid.n_img_per_temp_list_final = [1, 3, 3, 3]
cfg.save.grid.temp_list = [0.0, 0.6, 0.8, 1.0]
cfg.save.grid.border_pixels = 1

cfg.nn.encComb = 'add'  # 'add', 'None', 'maxAg', 'multiAdd'

cfg.nn.vae = False  # Set True for normal CVAE with just one latent space
cfg.nn.noDist = False  # Set True for FCN
cfg.nn.skip_connection = False  # False for FusionVAE, True for CVAE/FCN if skip connections are desired

cfg.temp.train = 1.0
cfg.temp.test = 1.0

# parameters of the discretized logistic mixture likelihood
cfg.mix.n_out_channels = 10
cfg.mix.out_channel_size = 10

cfg.val.freq = 20  # evaluate every n epochs
cfg.val.epochs = [0, 1, 5, 10]  # evaluate also after these epochs


def update_and_save_config():
    if cfg.args.dataset.startswith('mnist'):
        cfg.args.data = os.path.join(os.getenv('DIR_DATA'), 'mnist')
        cfg.ds.n_classes = 10
        cfg.ds.res = 32  # 28x28 with 2 pixels padding on each side
        cfg.ds.n_channels = 1
        cfg.ds.train_length = 60000
        cfg.ds.test_length = cfg.ds.valid_length = 10000
    elif cfg.args.dataset == 'celeba_64':  # 64x64
        cfg.ds.n_channels = 3
        cfg.ds.train.cv2 = True
        cfg.ds.lmdb.train = os.path.join(os.getenv('DIR_DATA'), 'celeba', 'celebA64_4NoiEllips_PNG', 'train.lmdb')
        cfg.ds.res = 64
        cfg.ds.n_classes = 40
        cfg.ds.train_length = 162770
        cfg.ds.test_length = 19962
        cfg.ds.valid_length = 19867
        cfg.ds.valid.multi_img = 4  # dataset contains multiple images in a special shape e.g. 256x64 i.e. 4 imgs
        cfg.ds.lmdb.valid = os.path.join(os.getenv('DIR_DATA'), 'celeba', 'celebA64_4NoiEllips_PNG', 'valid.lmdb')
        cfg.ds.valid.cv2 = True
    elif cfg.args.dataset == 'tless':  # 64x64
        cfg.ds.n_channels = 1
        cfg.ds.res = 64
        cfg.ds.n_classes = 21
        cfg.ds.train_length = 28009  # 20
        cfg.ds.test_length = cfg.ds.valid_length = 2579  # 20
        cfg.args.data = os.path.join(os.getenv('DIR_DATA'), 'blender', 'tless64occScaled')

        cfg.ds.lmdb.occ = os.path.join(os.getenv('DIR_DATA'), 'blender', 'tless64occScaledLive', 'train100x100_overlay_imgs_12classes.lmdb')
        cfg.ds.occ_length = 15552
        cfg.ds.multi_image_encoding = 3  # x noisy images and 1 target image are encoded together

        cfg.ds.overlay_img_res = 50  # 50x50
        cfg.aug.overlay_imgs = True
        cfg.aug.rot = True
        cfg.aug.move = 4  # max translation augmentation in each direction
        cfg.aug.scale = 4  # max scaling/shrinking
        cfg.aug.n_occ = (5, 8)  # min and max number of occlusions
        #                    01   02   03   04  05  06  07 08 09   10   11   12   13   14   15   16  17  18  19  20   21   22  23   24  25  26  27  28  29   30
        cfg.img.crop_val = [140, 135, 125, 125, 95, 95, 0, 0, 50, 100, 120, 110, 130, 120, 130, 130, 80, 90, 95, 95, 100, 100, 60, 110, 90, 90, 50, 70, 70, 110]

        cfg.ds.lmdb.train = os.path.join(os.getenv('DIR_DATA'), 'blender', 'tless64occScaledLive', 'train128x128cropped5socketClasses.lmdb')
        cfg.ds.train_length = 5836

        cfg.ds.lmdb.valid = os.path.join(os.getenv('DIR_DATA'), 'blender', 'tless64occScaled', 'valid64x64_augOcc_5socketClasses.lmdb')
        cfg.ds.test_length = cfg.ds.valid_length = 644
    else:
        raise NotImplementedError('Dataset %s is unknown!' % cfg.args.dataset)

    if cfg.args.dataset == 'mnist':
        cfg.noise.n_ellipses = 2
        cfg.noise.radius.low = 5
        cfg.noise.radius.high = 10
        cfg.noise.gaussian_var = 0.25
    else:
        cfg.noise.n_ellipses = 4
        cfg.noise.radius.low = 0
        cfg.noise.radius.high = 15
        cfg.noise.gaussian_var = 0

    cfg.val.n_samples = 10
    cfg.val.n_samples_final = 1000 if cfg.args.dataset == 'mnist' else 100

    cfg.args.res_dist = False

    cfg.n_noisy = cfg.args.n_noisy  # cfg.n_noisy is a variable that can change during training
    cfg.val.n_noisy_ind = True  # evaluate on 0..n_noisy images independently

    # Select aggregation method
    cfg.nn.addY_toLatQ = True  # aggregate only the features depending on y for using it to create q
    cfg.nn.encComb = 'multiAdd'  # 'add', 'None', 'multiAdd', agg, concat
    cfg.lat1.agg = 'maxAg'  # maxAg, meaAg, BayAg + convPBefAg
    cfg.lat1.convFBefAg = False
    cfg.lat1.convPBefAg = False

    cfg.lat.agg = 'maxAg'
    cfg.lat.convFBefAg = False
    cfg.lat.convPBefAg = False

    cfg.nn.skip_agg = 'maxAg'  # maxAg maxAgAdd maxAgConcat meaAg

    assert not (cfg.lat1.convFBefAg and cfg.lat1.convPBefAg)
    assert not (cfg.lat.convFBefAg and cfg.lat.convPBefAg)

    if cfg.save.grid.target_input_alone and cfg.save.grid.target_input_too:
        cfg.save.grid.target_input_too = False  # save prediction using just the target input image only once not twice

    if cfg.args.debug:
        cfg.misc.n_workers_train = 0
        cfg.misc.n_workers_valid = 0

        print("\nWarning: Debug parameters are used!\n")
        cfg.deb.train_iterations = 6  # -1 or >=1  # 10
        cfg.deb.test_iterations = 6  # -1 or >=1  # 10
        cfg.val.n_samples = 10  # 10
        cfg.val.n_samples_final = 10  # 10
        cfg.save.model = False
        cfg.save.train_tb = 1  # cfg.deb.train_iterations
        cfg.save.rec_img = 1  # cfg.deb.train_iterations
    cfg.save.gen_imgs = False

    if cfg.args.save:
        # save config
        output_file = os.path.join(cfg.args.save, 'cfg.yml')
        with open(output_file, 'w') as outfile:
            yaml.dump(cfg.to_dict(), outfile, default_flow_style=False)

        print("after save, cfg.lat.agg:", cfg.lat.agg)

