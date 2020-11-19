import torch

def get_test_patches(img, crop_size, stride_size):
    test_imgs = paint_border(img, crop_size, stride_size)
    test_img_patch = extract_patches(test_imgs, crop_size, stride_size)

    return test_img_patch, test_imgs.size(2), test_imgs.size(3)


def extract_patches(full_imgs, crop_size, stride_size):
    patch_height = crop_size
    patch_width = crop_size
    stride_height = stride_size
    stride_width = stride_size

    img_num, img_c, img_h, img_w = full_imgs.size()  # 4D arrays

    assert ((img_h - patch_height) % stride_height == 0 and (img_w - patch_width) % stride_width == 0)
    N_patches_img = ((img_h - patch_height) // stride_height + 1) * (
            (img_w - patch_width) // stride_width + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * img_num

    patches = torch.zeros((N_patches_tot, img_c, patch_height, patch_width))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(img_num):  # loop over the full images
        for h in range((img_h - patch_height) // stride_height + 1):
            for w in range((img_w - patch_width) // stride_width + 1):
                patch = full_imgs[i, :, h * stride_height:(h * stride_height) + patch_height,
                        w * stride_width:(w * stride_width) + patch_width]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches


def paint_border(imgs, crop_size, stride_size):
    patch_height = crop_size
    patch_width = crop_size
    stride_height = stride_size
    stride_width = stride_size

    img_num, img_c, img_h, img_w = imgs.size()
    leftover_h = (patch_height - img_h) % stride_height  # leftover on the h dim
    leftover_w = (patch_width - img_w) % stride_width  # leftover on the w dim
    full_imgs = torch.zeros((img_num, img_c, img_h+leftover_h, img_w+leftover_w))
    full_imgs[:, :, :img_h, :img_w] = imgs
    return full_imgs


def pred_to_patches(pred, crop_size, stride_size):
    return pred
    patch_height = crop_size
    patch_width = crop_size

    seg_num = 0
    #     print(pred.shape)

    assert (len(pred.shape) == 3)  # 3D array: (Npatches,height*width,2)

    pred_images = np.empty((pred.shape[0], pred.shape[1], seg_num + 1))  # (Npatches,height*width)
    pred_images[:, :, 0:seg_num + 1] = pred[:, :, 0:seg_num + 1]
    pred_images = np.reshape(pred_images, (pred_images.shape[0], patch_height, patch_width, seg_num + 1))
    return pred_images


def recompone_overlap(preds, crop_size, stride_size, img_h, img_w):
    assert (len(preds.size()) == 4)  # 4D arrays

    patch_h = crop_size
    patch_w = crop_size
    stride_height = stride_size
    stride_width = stride_size

    N_patches_h = (img_h - patch_h) // stride_height + 1
    N_patches_w = (img_w - patch_w) // stride_width + 1
    N_patches_img = N_patches_h * N_patches_w
    #     print("N_patches_h: " +str(N_patches_h))
    #     print("N_patches_w: " +str(N_patches_w))
    #     print("N_patches_img: " +str(N_patches_img))
    # assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.size(0) // N_patches_img
    img_c = preds.size(1)
    #     print("According to the dimension inserted, there are " +str(N_full_imgs) +" full images (of " +str(img_h)+"x" +str(img_w) +" each)")
    full_prob = torch.zeros((N_full_imgs, img_c, img_h, img_w))  # initialize to zero mega array with sum of Probabilities
    full_sum = torch.zeros((N_full_imgs, img_c, img_h, img_w))

    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_height + 1):
            for w in range((img_w - patch_w) // stride_width + 1):
                full_prob[i, :, h * stride_height:(h * stride_height) + patch_h,
                w * stride_width:(w * stride_width) + patch_w] += preds[k]
                full_sum[i, :, h * stride_height:(h * stride_height) + patch_h,
                w * stride_width:(w * stride_width) + patch_w] += 1
                k += 1
    #     print(k,preds.shape[0])
    assert (k == preds.size(0)), f'k: {k}, pred size[0]: {preds.size(0)}'
    assert (torch.sum(full_sum) >= 1.0)  # at least one
    final_avg = full_prob / full_sum
    #     print('using avg')
    return final_avg
