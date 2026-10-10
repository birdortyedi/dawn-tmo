"""Dawn: a tone mapping operator for multi-illuminant and low-light scenarios.

F. Kinli, B. Ozcan, F. Kirac, VISAPP 2025, pp. 62-68.

The scaling parameter is inferred from two luminance statistics that are invariant
to the arbitrary scale of an HDR file, so tone mapping the same scene at a
different exposure returns the same image. See FIXES.md for the measurements
behind each choice.
"""
import imageio.v2 as iio
import numpy as np
import cv2

GAMMA = 1.0 / 2.2
K_LINEAR = (0.25, 0.25, 2.0)     # (k1, k2, k3) for the linear scaling
K_NONLINEAR = (0.25, 2.0)        # (k1, k2) for the nonlinear scaling


def luminance(image):
    """Per-pixel luminance v = max(R, G, B), floored relative to the image.

    An absolute floor such as 1e-9 maps a black pixel to log(1e-9) = -20.7, which
    the log-domain box filter then spreads over a large neighbourhood.
    """
    v = np.max(np.maximum(np.asarray(image, np.float64), 0.0), axis=2)
    positive = v[v > 0]
    return np.maximum(v, 1e-6 * np.median(positive) if positive.size else 1e-9)


def image_statistics(image):
    """Scale-invariant statistics of the luminance: (Lw, mean_L, var_L).

    Lw      geometric mean of v, the image key
    mean_L  log(mean(v) / Lw), the log arithmetic-to-geometric ratio
    var_L   std(log v), the luminance spread in nats

    Both statistics are unchanged by v -> c*v, which is what makes the operator
    invariant to the exposure the HDR file happens to be stored at.
    """
    log_v = np.log(luminance(image))
    Lw = float(np.exp(log_v.mean()))
    return Lw, float(np.log(np.exp(log_v).mean() / Lw)), float(log_v.std())


def adaptive_scaling_parameter(image, k1=None, k2=None, k3=None):
    k1, k2, k3 = (K_LINEAR[0] if k1 is None else k1,
                  K_LINEAR[1] if k2 is None else k2,
                  K_LINEAR[2] if k3 is None else k3)
    _, mean_L, var_L = image_statistics(image)
    return k1 * mean_L + k2 * var_L + k3


def nonlinear_scaling_parameter(image, k1=None, k2=None):
    k1 = K_NONLINEAR[0] if k1 is None else k1
    k2 = K_NONLINEAR[1] if k2 is None else k2
    _, mean_L, var_L = image_statistics(image)
    return np.exp(k1 * mean_L) + k2 * np.log1p(var_L)


def _box(n):
    """Odd box width of at least 3, so the window is centred on its own pixel."""
    n = int(max(3, n))
    return n if n % 2 else n + 1


def apply_dawn(hdr, kernels=(1, 4), k1=None, k2=None, k3=None, nonlinear_scaling=False):
    """Tone map an HDR RGB array to display values in [0, 1].

    kernels : each entry is one scale of the image key, averaged at the end.
              1 is the global geometric mean of Eq. 1; n > 1 is a local key over
              a min(rows, cols) // n window. (1,) alone gives the global operator.

    k1, k2, k3 default per branch, so supplying one constant never replaces another.
    """
    hdr = np.maximum(np.asarray(hdr, np.float64), 0.0)
    rows, cols, _ = hdr.shape
    a = (nonlinear_scaling_parameter(hdr, k1, k2) if nonlinear_scaling
         else adaptive_scaling_parameter(hdr, k1, k2, k3))
    v = luminance(hdr)
    lv = np.log(v)
    Lw = np.exp(lv.mean())
    keys = [np.full_like(v, Lw) if kernel == 1
            else np.exp(cv2.boxFilter(lv, -1, (_box(min(rows, cols) // kernel),) * 2))
            for kernel in kernels]
    ldr = sum(hdr / (v + a * key)[..., None] for key in keys) / len(keys)
    return np.clip(ldr, 0., 1.) ** GAMMA      # clip before gamma: no NaN from negatives


def to_uint8(ldr):
    """imageio no longer writes float arrays to PNG."""
    return np.round(np.clip(ldr, 0., 1.) * 255.).astype(np.uint8)


if __name__ == '__main__':
    im_f64 = iio.imread("hdr-images/Desk_oBA2.hdr", format="HDR-FI")
    im = im_f64.astype(np.float32)
    ldr = apply_dawn(im, nonlinear_scaling=True)
    iio.imwrite("ldr-outputs/Desk_oBA2.png", to_uint8(ldr))
