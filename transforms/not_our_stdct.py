"""DCT Implementation by jonashaag on GitHub https://github.com/jonashaag/pydct/tree/master"""
import torch
import torch_dct


def sdct_torch(signals, frame_length, frame_step, window=torch.hamming_window):
    """Compute Short-Time Discrete Cosine Transform of `signals`.

    No padding is applied to the signals.

    Parameters
    ----------
    signal : Time-domain input signal(s), a `[..., n_samples]` tensor.

    frame_length : Window length and DCT frame length in samples.

    frame_step : Number of samples between adjacent DCT columns.

    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.

    Returns
    -------
    dct : Real-valued F-T domain DCT matrix/matrixes, a `[..., frame_length, n_frames]` tensor.
    """
    framed = signals.unfold(-1, frame_length, frame_step)
    if callable(window):
        window = window(frame_length).to(framed)
    if window is not None:
        framed = framed * window
    return torch_dct.dct(framed, norm=None).transpose(-1, -2)


def isdct_torch(dcts, *, frame_step, frame_length=None, window=torch.hamming_window):
    """Compute Inverse Short-Time Discrete Cosine Transform of `dct`.

    Parameters other than `dcts` are keyword-only.

    Parameters
    ----------
    dcts : DCT matrix/matrices from `sdct_torch`

    frame_step : Number of samples between adjacent DCT columns (should be the
        same value that was passed to `sdct_torch`).

    frame_length : Ignored.  Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct_torch`.

    window : Window to use for DCT.  Either a window tensor (see documentation for `torch.stft`),
        or a window tensor constructor, `window(frame_length) -> Tensor`.
        Default: hamming window.

    Returns
    -------
    signals : Time-domain signal(s) reconstructed from `dcts`, a `[..., n_samples]` tensor.
        Note that `n_samples` may be different from the original signals' lengths as passed to `sdct_torch`,
        because no padding is applied.
    """
    *_, frame_length2, n_frames = dcts.shape
    assert frame_length in {None, frame_length2}
    signals = torch_overlap_add(
        torch_dct.idct(dcts.transpose(-1, -2), norm=None).transpose(-1, -2),
        frame_step=frame_step,
    )
    if callable(window):
        window = window(frame_length2).to(signals)
    if window is not None:
        window_frames = window[:, None].expand(-1, n_frames)
        window_signal = torch_overlap_add(window_frames, frame_step=frame_step)
        signals = signals / window_signal
    return signals


def torch_overlap_add(framed, *, frame_step, frame_length=None):
    """Overlap-add ("deframe") a framed signal.

    Parameters other than `framed` are keyword-only.

    Parameters
    ----------
    framed : Tensor of shape `(..., frame_length, n_frames)`.

    frame_step : Overlap to use when adding frames.

    frame_length : Ignored.  Window length and DCT frame length in samples.
        Can be None (default) or same value as passed to `sdct_torch`.

    Returns
    -------
    deframed : Overlap-add ("deframed") signal.
        Tensor of shape `(..., (n_frames - 1) * frame_step + frame_length)`.
    """
    *rest, frame_length2, n_frames = framed.shape
    assert frame_length in {None, frame_length2}
    return torch.nn.functional.fold(
        framed.reshape(-1, frame_length2, n_frames),
        output_size=(((n_frames - 1) * frame_step + frame_length2), 1),
        kernel_size=(frame_length2, 1),
        stride=(frame_step, 1),
    ).reshape(*rest, -1)