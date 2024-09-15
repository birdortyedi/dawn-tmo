import imageio as iio
import numpy as np
import cv2


def adaptive_scaling_parameter(image, k1=0.5, k2=0.5, k3=1.0):
    mean_L, var_L = np.mean(image), np.var(image)
    a = k1 * mean_L + k2 * var_L + k3
    return a


def nonlinear_scaling_parameter(image, k1=0.5, k2=0.2):
    mean_L, var_L = np.mean(image), np.var(image)
    a = np.exp(k1 * mean_L) + k2 * np.log(1 + var_L)
    return a


def apply_dawn(hdr, kernels=(1, 4), k1=0.5, k2=0.5, k3=1.0, nonlinear_scaling=False):
    a = nonlinear_scaling_parameter(hdr, k1, k2) if nonlinear_scaling else adaptive_scaling_parameter(hdr, k1, k2, k3) 
    rows, cols, _ = hdr.shape
    v = np.max(hdr, axis=2) 
    v[v == 0] = 1e-9
    lv = np.log(v)
    ldr = sum([
        hdr / np.tile(
            np.expand_dims(
                a * np.exp(cv2.boxFilter(lv, -1, (int(min(rows // kernel, cols // kernel)),) * 2)) + v, axis=2),
                (1, 1, 3)
            ) 
        for kernel in kernels
    ]) / len(kernels)
    ldr **= 1.0 / 2.2
    ldr = np.clip(ldr, a_min=0., a_max=1.)
    return ldr



if __name__ == '__main__':
    im_f64 = iio.imread("hdr-images/Desk_oBA2.hdr", format="HDR-FI")
    im = im_f64.astype(np.float32)
    ldr = apply_dawn(im, nonlinear_scaling=True)
    iio.imwrite("ldr-outputs/Desk_oBA2.png", ldr)
