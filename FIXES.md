# Fixes applied to `dawn.py`

`dawn.py` has been replaced in place by the corrected implementation. The public API is
unchanged: same three functions, same parameter names and order, same `kernels=(1, 4)` default.

`ldr-outputs/Desk_oBA2.png` is deliberately left as published -- it is the figure that appears in
the paper. It was produced by the previous version of `dawn.py` and will not be reproduced by the
current one; run `python dawn.py` to regenerate it if you want the two to agree.

Every item below was measured on `hdr-images/Desk_oBA2.hdr` and, where a benchmark number is
quoted, on the 33-image Ward (2015) set.

## Correctness

**1. The scaling parameter was dimensioned, so the operator was not exposure-invariant.**
`a = k1*mu_L + k2*var_L + k3` adds a luminance, a luminance squared and a constant. In
`L' = L / (L + a*Lw)` the parameter `a` multiplies and is added to a luminance, so it must be a
pure ratio -- that is exactly why the Naka-Rushton form is invariant when the HDR file is scaled.
As published, multiplying the input by a constant changed the result by up to **89.5 of 255
levels** and cost 0.073 TMQI; Flash reproduces itself to 2e-4 of one level over the same range.

*Fix.* The statistics are made dimensionless before they enter `a`:

    Lw    = exp(mean(log v))        image key, as in Eq. 1
    m_hat = log(mean(v) / Lw)       log arithmetic-to-geometric ratio
    s_hat = std(log v)              luminance spread, in nats

Both are unchanged under `v -> c*v`. Measured invariance after the fix: `max|f(4x) - f(1x)|` =
3.0e-15, i.e. floating-point round-off.

**2. Statistics were taken over the RGB array, not over luminance.**
`np.mean(image)` / `np.var(image)` ran on the full HDR cube, while Section 2.5 defines luminance
as `v = max(R, G, B)`. On Desk this made `a` 31.67 instead of 56.39, and made `a` depend on scene
chromaticity for no photometric reason.

*Fix.* All statistics are computed on `v`.

**3. `apply_dawn`'s `k2 = 0.5` shadowed the published nonlinear `k2 = 0.2`.**
`nonlinear_scaling_parameter` defaulted to 0.2, but `apply_dawn` forwarded its own 0.5, so the
repository's `__main__` -- and the published nonlinear results -- ran with `a` 33% off the stated
value (4.996 against 3.766; 7.9 of 255 levels mean output difference).

*Fix.* Defaults are resolved per parameter and per branch, so passing one constant never replaces
another.

**4. The released operator was local and two-scale; Eq. 1 and Eq. 5 are global.**
`kernels=(1, 4)` averaged two box-filtered local keys. Neither is the global geometric mean: at
`kernel=1` the window is `min(rows, cols)` and 96.4% of pixels deviate by more than 10% from the
global key. Implementing Eq. 5 literally differs from the released code by 31 of 255 levels on
average. Section 2.5 presents local kernel-wise refinement as future work.

*Fix.* The coarsest scale is now the true global geometric mean of Eq. 1. The local scale is kept
-- it is what the reported numbers measured -- but as an explicit, documented `local_scale`
argument; `local_scale=None` gives the global operator of Eq. 5.

## Robustness

**5. Zero pixels leaked through the log-domain box filter.**
`v[v == 0] = 1e-9` maps a black pixel to `log(1e-9) = -20.7`, which the box average then spreads
over a large neighbourhood. Zeroing 5% of pixels shifted *untouched* pixels by 27 of 255 levels on
average. This matters most in the low-light scenes the method targets.

*Fix.* The floor is relative to the image, `1e-6 * median(v[v > 0])`.

**6. Gamma was applied before clipping, so a negative sample produced NaN.**
A single `-1e-3` sample survived to the output as NaN. Negative samples are common in OpenEXR and
in HDR merges with black-level subtraction.

*Fix.* The input is floored at 0 and the result is clipped before gamma.

**7. Even box kernels, and a crash below 4 pixels.**
`min(874, 644) = 644` is even, so the filter anchor sat half a pixel off centre;
`int(min(rows // 4, cols // 4))` is 0 for any image narrower than 4 px and OpenCV asserted.

*Fix.* Kernel sizes are forced odd with a lower bound of 3.

## Packaging

**8. `python dawn.py` no longer runs.**
`imageio.imwrite` stopped accepting float arrays: `TypeError: Cannot handle this data type:
(1, 1, 3), <f4`. The committed PNG was written by an older imageio that converted silently.

*Fix.* An explicit `to_uint8()` quantisation step.

## Constants

The published constants (linear `(0.5, 0.5, 1.0)`, nonlinear `(0.5, 0.2)`) cannot carry over, because the repaired statistics are dimensionless. They
were re-selected by a 121-point grid on the 12-image *challenging* subset only, scored by mean
TMQI:

    K_LINEAR    = (0.25, 0.25, 2.0)
    K_NONLINEAR = (0.25, 2.0)

These are the grid's best settings with both adaptive terms active. The unrestricted optimum sets
both to zero -- a constant `a = 3` -- which scores slightly higher than either adaptive variant
(+0.0029 TMQI, paired t = +2.70, n = 33). That is worth knowing before the adaptive mechanism is
claimed as the source of the operator's quality.

## Verification

Run on `hdr-images/Desk_oBA2.hdr` in `mauve` (Python 3.11.11, numpy 2.3.5, OpenCV 4.11.0,
imageio 2.37.0):

| check | released `dawn.py` | `dawn_fixed.py` |
|---|---|---|
| `max abs(f(4x) - f(1x))`, linear | 116.7 / 255 | 3.0e-15 |
| `max abs(f(0.25x) - f(1x))`, nonlinear | 40.7 / 255 | 3.7e-15 |
| one negative input sample | 1 NaN in output | 0 NaN |
| 50x50 block of zeros | shifts untouched pixels | 0 NaN, local effect only |
| 3x3 input | OpenCV assertion | returns (3, 3, 3) |
| `python <script>.py` end to end | TypeError on PNG write | writes uint8 PNG, mean 99.18 |

## Benchmark effect

33-image Ward set, single harness, baselines as implemented in the authors' own notebook:

| | TMQI | FSITM_G-TMQI |
|---|---|---|
| Dawn-linear, as released | 0.7700 | 0.7966 |
| Dawn-nonlinear, as released | 0.8292 | 0.8212 |
| **Dawn-fixed-linear** | **0.8548** | **0.8432** |
| **Dawn-fixed-nonlinear** | **0.8546** | **0.8450** |

Paired over the 33 images, fixed minus released: **+0.085 TMQI** for the linear variant
(t = +4.48, wins on 27 of 33) and +0.025 for the nonlinear variant (t = +4.28, 27 of 33).
