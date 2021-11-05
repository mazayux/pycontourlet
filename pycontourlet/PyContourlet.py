# -*- coding: utf-8 -*-
#    PyContourlet
#
#    A Python library for the Contourlet Transform.
#
#    Copyright (C) 2011 Mazay Jiménez
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation version 2.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from numpy import *
from scipy import weave, signal
from scipy.fftpack import fftshift
from scipy.weave import converters
from matplotlib.cbook import is_string_like
import pdb
# Laplacian Pyramid


def lpdec(x, h, g):
    """ LPDEC   Laplacian Pyramid Decomposition

    [c, d] = lpdec(x, h, g)

    Input:
    x:      input image
    h, g:   two lowpass filters for the Laplacian pyramid

    Output:
    c:      coarse image at half size
    d:      detail image at full size

    See also:   LPREC, PDFBDEC"""

    # Lowpass filter and downsample
    xlo = sefilter2(x, h, h, 'per')
    c = xlo[::2, ::2]

    # Compute the residual (bandpass) image by upsample, filter, and subtract
    # Even size filter needs to be adjusted to obtain perfect reconstruction
    adjust = mod(size(g) + 1, 2)

    xlo = zeros(x.shape)
    xlo[::2, ::2] = c
    d = x - sefilter2(xlo, g, g, 'per', adjust * array([1, 1]))

    return c, d


def lprec(c, d, h, g):
    """ LPDEC   Laplacian Pyramid Reconstruction

    x = lprec(c, d, h, g)

    Input:
    c:      coarse image at half size
    d:      detail image at full size
    h, g:   two lowpass filters for the Laplacian pyramid

    Output:
    x:      reconstructed image

    Note:     This uses a new reconstruction method by Do and Vetterli,
    Framming pyramids, IEEE Trans. on Sig Proc., Sep. 2003.

    See also:   LPDEC, PDFBREC"""

    # First, filter and downsample the detail image
    xhi = sefilter2(d, h, h, 'per')
    xhi = xhi[::2, ::2]

    # Subtract from the coarse image, and then upsample and filter
    xlo = c - xhi
    xlo = dup(xlo, array([2, 2]))

    # Even size filter needs to be adjusted to obtain
    # perfect reconstruction with zero shift
    adjust = mod(size(g) + 1, 2)

    xlo = sefilter2(xlo, g, g, 'per', adjust * array([1, 1]))

    # Final combination
    x = xlo + d

    return x

# Wavelet Filter Bank


def wfb2dec(x, h, g):
    """% WFB2DEC   2-D Wavelet Filter Bank Decomposition
    %
    %       y = wfb2dec(x, h, g)
    %
    % Input:
    %   x:      input image
    %   h, g:   lowpass analysis and synthesis wavelet filters
    %
    % Output:
    %   x_LL, x_LH, x_HL, x_HH:   Four 2-D wavelet subbands"""

    # Make sure filter in a row vector
    h = h.flatten(1)[:, newaxis].T
    g = g.flatten(1)[:, newaxis].T

    h0 = h.copy()
    len_h0 = size(h0)
    ext_h0 = floor(len_h0 / 2.0)
    # Highpass analysis filter: H1(z) = -z^(-1) G0(-z)
    len_h1 = size(g)
    c = floor((len_h1 + 1.0) / 2.0)
    # Shift the center of the filter by 1 if its length is even.
    if mod(len_h1, 2) == 0:
        c = c + 1
    h1 = - g * (-1)**(arange(1, len_h1 + 1) - c)
    ext_h1 = len_h1 - c + 1

    # Row-wise filtering
    x_L = rowfiltering(x, h0, ext_h0)
    x_L = x_L[:, ::2]

    x_H = rowfiltering(x, h1, ext_h1)
    x_H = x_H[:, ::2]

    # Column-wise filtering
    x_LL = rowfiltering(x_L.T, h0, ext_h0).T
    x_LL = x_LL[::2, :]

    x_LH = rowfiltering(x_L.T, h1, ext_h1).T
    x_LH = x_LH[::2, :]

    x_HL = rowfiltering(x_H.T, h0, ext_h0).T
    x_HL = x_HL[::2, :]

    x_HH = rowfiltering(x_H.T, h1, ext_h1).T
    x_HH = x_HH[::2, :]

    return x_LL, x_LH, x_HL, x_HH


def wfb2rec(x_LL, x_LH, x_HL, x_HH, h, g):
    """% WFB2REC   2-D Wavelet Filter Bank Decomposition
    %
    %       x = wfb2rec(x_LL, x_LH, x_HL, x_HH, h, g)
    %
    % Input:
    %   x_LL, x_LH, x_HL, x_HH:   Four 2-D wavelet subbands
    %   h, g:   lowpass analysis and synthesis wavelet filters
    %
    % Output:
    %   x:      reconstructed image"""

    # Make sure filter in a row vector
    h = h.flatten(1)[:, newaxis].T
    g = g.flatten(1)[:, newaxis].T

    g0 = g.copy()
    len_g0 = size(g0)
    ext_g0 = floor((len_g0 - 1) / 2)

    # Highpass synthesis filter: G1(z) = -z H0(-z)
    len_g1 = size(h)
    c = floor((len_g1 + 1) / 2)
    g1 = (-1) * h * (-1) ** (arange(1, len_g1 + 1) - c)
    ext_g1 = len_g1 - (c + 1)

    # Get the output image size
    height, width = x_LL.shape
    x_B = zeros((height * 2, width))
    x_B[::2, :] = x_LL

    # Column-wise filtering
    x_L = rowfiltering(x_B.T, g0, ext_g0).T
    x_B[::2, :] = x_LH
    x_L = x_L + rowfiltering(x_B.T, g1, ext_g1).T

    x_B[::2, :] = x_HL
    x_H = rowfiltering(x_B.T, g0, ext_g0).T
    x_B[::2, :] = x_HH
    x_H = x_H + rowfiltering(x_B.T, g1, ext_g1).T

    # Row-wise filtering
    x_B = zeros((2 * height, 2 * width))
    x_B[:, ::2] = x_L
    x = rowfiltering(x_B, g0, ext_g0)
    x_B[:, ::2] = x_H
    x = x + rowfiltering(x_B, g1, ext_g1)

    return x

# Internal function: Row-wise filtering with
# border handling (used in wfb2dec and wfb2rec)


def rowfiltering(x, f, ext1):
    ext2 = size(f) - ext1 - 1
    x = c_[x[:, -ext1::], x, x[:, 0:ext2]]
    y = signal.convolve(x, f, 'valid')
    return y

# Directional Filter Bank


def dfbdec(x, fname, n):
    """ DFBDEC   Directional Filterbank Decomposition

    y = dfbdec(x, fname, n)

    Input:
    x:      input image
    fname:  filter name to be called by DFILTERS
    n:      number of decomposition tree levels

    Output:
    y:      subband images in a cell vector of length 2^n

    Note:
    This is the general version that works with any FIR filters

    See also: DFBREC, FBDEC, DFILTERS"""

    if (n != round(n)) or (n < 0):
        print 'Number of decomposition levels must be a non-negative integer'

    if n == 0:
        # No decomposition, simply copy input to output
        y = [None]
        y[0] = x.copy()
        return y

    # Get the diamond-shaped filters
    h0, h1 = dfilters(fname, 'd')

    # Fan filters for the first two levels
    # k0: filters the first dimension (row)
    # k1: filters the second dimension (column)
    k0 = modulate2(h0, 'c')
    k1 = modulate2(h1, 'c')
    # Tree-structured filter banks
    if n == 1:
        # Simplest case, one level
        y = [[None]] * 2
        y[0], y[1] = fbdec(x, k0, k1, 'q', '1r', 'per')
    else:
        # For the cases that n >= 2
        # First level
        x0, x1 = fbdec(x, k0, k1, 'q', '1r', 'per')
        # Second level
        y = [[None]] * 4
        y[0], y[1] = fbdec(x0, k0, k1, 'q', '2c', 'qper_col')
        y[2], y[3] = fbdec(x1, k0, k1, 'q', '2c', 'qper_col')
        # Fan filters from diamond filters
        f0, f1 = ffilters(h0, h1)
        # Now expand the rest of the tree
        for l in xrange(3, n + 1):
            # Allocate space for the new subband outputs
            y_old = y[:]
            y = [[None]] * 2**l
            # The first half channels use R1 and R2
            for k in xrange(0, 2**(l - 2)):
                i = mod(k, 2)
                y[2 * k], y[2 * k + 1] = fbdec(y_old[k],
                                               f0[i], f1[i], 'pq', i, 'per')
            # The second half channels use R3 and R4
            for k in xrange(2**(l - 2), 2**(l - 1)):
                i = mod(k, 2) + 2
                y[2 * k], y[2 * k + 1] = fbdec(y_old[k],
                                               f0[i], f1[i], 'pq', i, 'per')
    # Back sampling (so that the overal sampling is separable)
    # to enhance visualization
    y = backsamp(y)
    # Flip the order of the second half channels
    y[2**(n - 1)::] = y[::-1][:2**(n - 1)]

    return y


def dfbrec(y, fname):
    """ DFBREC   Directional Filterbank Reconstruction

    x = dfbrec(y, fname)

    Input:
    y:      subband images in a cell vector of length 2^n
    fname:  filter name to be called by DFILTERS

    Output:
    x:      reconstructed image

    See also: DFBDEC, FBREC, DFILTERS"""

    n = int(log2(len(y)))

    if (n != round(n)) or (n < 0):
        print 'Number of reconstruction levels must be a non-negative integer'

    if n == 0:
        # Simply copy input to output
        x = [None]
        x[0] = y[0][:]
        return x

    # Get the diamond-shaped filters
    h0, h1 = dfilters(fname, 'r')

    # Fan filters for the first two levels
    # k0: filters the first dimension (row)
    # k1: filters the second dimension (column)
    k0 = modulate2(h0, 'c')
    k1 = modulate2(h1, 'c')

    # Flip back the order of the second half channels
    y[2**(n - 1)::] = y[::-1][:2**(n - 1)]

    # Undo backsampling
    y = rebacksamp(y)

    # Tree-structured filter banks

    if n == 1:
        # Simplest case, one level
        x = fbrec(y[0], y[1], k0, k1, 'q', '1r', 'per')
    else:
        # For the cases that n >= 2
        # Fan filters from diamond filters
        f0, f1 = ffilters(h0, h1)

        # Recombine subband outputs to the next level
        for l in xrange(n, 2, -1):
            y_old = y[:]
            y = [[None]] * 2**(l - 1)

            # The first half channels use R1 and R2
            for k in xrange(0, 2**(l - 2)):
                i = mod(k, 2)
                y[k] = fbrec(y_old[2 * k], y_old[2 * k + 1],
                             f0[i], f1[i], 'pq', i, 'per')
            # The second half channels use R3 and R4
            for k in xrange(2**(l - 2), 2**(l - 1)):
                i = mod(k, 2) + 2
                y[k] = fbrec(y_old[2 * k], y_old[2 * k + 1],
                             f0[i], f1[i], 'pq', i, 'per')

        # Second level
        x0 = fbrec(y[0], y[1], k0, k1, 'q', '2c', 'qper_col')
        x1 = fbrec(y[2], y[3], k0, k1, 'q', '2c', 'qper_col')

        # First level
        x = fbrec(x0, x1, k0, k1, 'q', '1r', 'per')

    return x


def dfbdec_l(x, f, n):
    """ DFBDEC_L   Directional Filterbank Decomposition using Ladder Structure

    y = dfbdec_l(x, f, n)

    Input:
    x:  input image
    f:  filter in the ladder network structure,
    can be a string naming a standard filter (see LDFILTER)
    n:  number of decomposition tree levels

    Output:
    y:  subband images in a cell array (of size 2^n x 1)"""

    if (n != round(n)) or (n < 0):
        print 'Number of decomposition levels must be a non-negative integer'

    if n == 0:
        # No decomposition, simply copy input to output
        y = [None]
        y[0] = x
        return y

    # Ladder filter
    if str(f) == f:
        f = ldfilter(f)

    # Tree-structured filter banks
    if n == 1:
        # Simplest case, one level
        y = [[None]] * 2
        y[0], y[1] = fbdec_l(x, f.copy(), 'q', '1r', 'qper_col')
    else:
        # For the cases that n >= 2
        # First level
        x0, x1 = fbdec_l(x, f.copy(), 'q', '1r', 'qper_col')

        # Second level
        y = [[None]] * 4
        y[1], y[0] = fbdec_l(x0, f.copy(), 'q', '2c', 'per')
        y[3], y[2] = fbdec_l(x1, f.copy(), 'q', '2c', 'per')

        # Now expand the rest of the tree
        for l in xrange(3, n + 1):
            # Allocate space for the new subband outputs
            y_old = y[:]
            y = [[None]] * 2**l

            # The first half channels use R1 and R2
            for k in xrange(0, 2**(l - 2)):
                i = mod(k, 2)
                y[2 * k + 1], y[2 *
                                k] = fbdec_l(y_old[k], f.copy(), 'p', i, 'per')

                # The second half channels use R3 and R4
            for k in xrange(2**(l - 2), 2**(l - 1)):
                i = mod(k, 2) + 2
                y[2 * k + 1], y[2 *
                                k] = fbdec_l(y_old[k], f.copy(), 'p', i, 'per')

    # Backsampling
    y = backsamp(y)

    # Flip the order of the second half channels
    y[2**(n - 1)::] = y[::-1][:2**(n - 1)]
    return y


def dfbrec_l(y, f):
    """ DFBREC_L   Directional Filterbank Reconstruction using Ladder Structure

    x = dfbrec_l(y, fname)

    Input:
    y:  subband images in a cell vector of length 2^n
    f:  filter in the ladder network structure,
    can be a string naming a standard filter (see LDFILTER)

    Output:
    x:  reconstructed image

    See also:   DFBDEC, FBREC, DFILTERS"""

    n = int(log2(len(y)))

    if (n != round(n)) or (n < 0):
        print 'Number of reconstruction levels must be a non-negative integer'

    if n == 0:
        # Simply copy input to output
        x = y[0][:]
        return x

    # Ladder filter
    if str(f) == f:
        f = ldfilter(f)

    # Flip back the order of the second half channels
    y[2**(n - 1)::] = y[::-1][:2**(n - 1)]
    # Undo backsampling
    y = rebacksamp(y)

    # Tree-structured filter banks
    if n == 1:
        # Simplest case, one level
        x = fbrec_l(y[0], y[1], f.copy(), 'q', '1r', 'qper_col')
    else:
        # For the cases that n >= 2
        # Recombine subband outputs to the next level
        for l in xrange(n, 2, -1):
            y_old = y[:]
            y = [[None]] * 2**(l - 1)

            # The first half channels use R0 and R1
            for k in xrange(0, 2**(l - 2)):
                i = mod(k, 2)
                y[k] = fbrec_l(y_old[2 * k + 1], y_old[2 * k],
                               f.copy(), 'p', i, 'per')
            # The second half channels use R2 and R3
            for k in xrange(2**(l - 2), 2**(l - 1)):
                i = mod(k, 2) + 2
                y[k] = fbrec_l(y_old[2 * k + 1], y_old[2 * k],
                               f.copy(), 'p', i, 'per')
        # Second level
        x0 = fbrec_l(y[1], y[0], f.copy(), 'q', '2c', 'per')
        x1 = fbrec_l(y[3], y[2], f.copy(), 'q', '2c', 'per')
        # First level
        x = fbrec_l(x0, x1, f.copy(), 'q', '1r', 'qper_col')

    return x

# Two-channel 2D filter banks (used in DFB)


def fbdec(x, h0, h1, type1, type2, extmod='per'):
    """ FBDEC   Two-channel 2D Filterbank Decomposition

    [y0, y1] = fbdec(x, h0, h1, type1, type2, [extmod])

    Input:
    x:  input image
    h0, h1: two decomposition 2D filters
    type1:  'q', 'p' or 'pq' for selecting quincunx or parallelogram
    downsampling matrix
    type2:  second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    If type1 == 'p' then type2 is one of {0, 1, 2, 3}
    Those are specified in QDOWN and PDOWN
    If type1 == 'pq' then same as 'p' except that
    the paralellogram matrix is replaced by a combination
    of a  resampling and a quincunx matrices
    extmod: [optional] extension mode (default is 'per')

    Output:
    y0, y1: two result subband images

    Note:       This is the general implementation of 2D two-channel
    filterbank

    See also:   FBDEC_SP """

    # For parallegoram filterbank using quincunx downsampling, resampling is
    # applied before filtering
    if type1 == 'pq':
        x = resamp(x, type2)

    # Stagger sampling if filter is odd-size (in both dimensions)
    if all(mod(h1.shape, 2)):
        shift = array([[-1], [0]])

        # Account for the resampling matrix in the parallegoram case
        if type1 == 'p':
            R = [[None]] * 4
            R[0] = array([[1, 1], [0, 1]])
            R[1] = array([[1, -1], [0, 1]])
            R[2] = array([[1, 0], [1, 1]])
            R[3] = array([[1, 0], [-1, 1]])
            shift = R[type2] * shift
    else:
        shift = array([[0], [0]])
    # Extend, filter and keep the original size
    y0 = efilter2(x, h0, extmod)
    y1 = efilter2(x, h1, extmod, shift)
    # Downsampling
    if type1 == 'q':
        # Quincunx downsampling
        y0 = qdown(y0, type2)
        y1 = qdown(y1, type2)
    elif type1 == 'p':
        # Parallelogram downsampling
        y0 = pdown(y0, type2)
        y1 = pdown(y1, type2)
    elif type1 == 'pq':
        # Quincux downsampling using the equipvalent type
        pqtype = ['1r', '2r', '2c', '1c']
        y0 = qdown(y0, pqtype[type2])
        y1 = qdown(y1, pqtype[type2])
    else:
        print 'Invalid input type1'
    return y0, y1


def fbrec(y0, y1, h0, h1, type1, type2, extmod='per'):
    """ FBREC   Two-channel 2D Filterbank Reconstruction

    x = fbrec(y0, y1, h0, h1, type1, type2, [extmod])

    Input:
    y0, y1: two input subband images
    h0, h1: two reconstruction 2D filters
    type1:  'q', 'p' or 'pq' for selecting quincunx or parallelogram
    upsampling matrix
    type2:  second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    If type1 == 'p' then type2 is one of {0, 1, 2, 3}
    Those are specified in QUP and PUP
    If type1 == 'pq' then same as 'p' except that
    the paralellogram matrix is replaced by a combination
    of a quincunx and a resampling matrices
    extmod: [optional] extension mode (default is 'per')

    Output:
    x:  reconstructed image

    Note:   This is the general case of 2D two-channel filterbank

    See also:   FBDEC"""

    # Upsampling
    if type1 == 'q':
        # Quincunx upsampling
        y0 = qup(y0, type2)
        y1 = qup(y1, type2)
    elif type1 == 'p':
        # Parallelogram upsampling
        y0 = pup(y0, type2)
        y1 = pup(y1, type2)
    elif type1 == 'pq':
        # Quincux upsampling using the equivalent type
        pqtype = ['1r', '2r', '2c', '1c']
        y0 = qup(y0, pqtype[type2])
        y1 = qup(y1, pqtype[type2])
    else:
        print 'Invalid input type1'

    # Stagger sampling if filter is odd-size
    if all(mod(h1.shape, 2)):
        shift = array([[1], [0]])
        # Account for the resampling matrix in the parallegoram case
        if type1 == 'p':
            R = [[None]] * 4
            R[0] = array([[1, 1], [0, 1]])
            R[1] = array([[1, -1], [0, 1]])
            R[2] = array([[1, 0], [1, 1]])
            R[3] = array([[1, 0], [-1, 1]])
            shift = R[type2] * shift
    else:
        shift = array([[0], [0]])

    # Dimension that has even size filter needs to be adjusted to obtain
    # perfect reconstruction with zero shift
    adjust0 = mod(array([h0.shape]) + 1, 2).T
    adjust1 = mod(array([h1.shape]) + 1, 2).T

    # Extend, filter and keep the original size
    x0 = efilter2(y0, h0, extmod, adjust0)
    x1 = efilter2(y1, h1, extmod, adjust1 + shift)

    # Combine 2 channel to output
    x = x0 + x1

    # For parallegoram filterbank using quincunx upsampling,
    # a resampling is required at the end
    if type1 == 'pq':
        # Inverse of resamp(x, type)
        inv_type = [1, 0, 3, 2]
        x = resamp(x, inv_type[type2])
    return x


def fbdec_l(x, f, type1, type2, extmod='per'):
    """ FBDEC_L   Two-channel 2D Filterbank Decomposition using Ladder Structure

    [y0, y1] = fbdec_l(x, f, type1, type2, [extmod])

    Input:
    x:  input image
    f:  filter in the ladder network structure
    type1:  'q' or 'p' for selecting quincunx or parallelogram
    downsampling matrix
    type2:  second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    ({1, 2, 0, 3} can also be used as equivalent)
    If type1 == 'p' then type2 is one of {0, 1, 2, 3}
    Those are specified in QPDEC and PPDEC
    extmod: [optional] extension mode (default is 'per')
    This refers to polyphase components.

    Output:
    y0, y1: two result subband images

    Note:       This is also called the lifting scheme

    See also:   FBDEC, FBREC_L"""

    # Modulate f
    f[:, ::2] = -f[:, ::2]

    if min(x.shape) == 1:
        print 'Input is a vector, unpredicted output!'

    # Polyphase decomposition of the input image
    if str.lower(type1[0]) == 'q':
        # Quincunx polyphase decomposition
        p0, p1 = qpdec(x, type2)
    elif str.lower(type1[0]) == 'p':
        # Parallelogram polyphase decomposition
        p0, p1 = ppdec(x, type2)
    else:
        print 'Invalid argument type1'

    # Ladder network structure
    y0 = (1 / sqrt(2)) * (p0 - sefilter2(p1, f, f, extmod, array([[1], [1]])))
    y1 = (-sqrt(2) * p1) - sefilter2(y0, f, f, extmod)

    return y0, y1


def fbrec_l(y0, y1, f, type1, type2, extmod='per'):
    """ FBREC_L   Two-channel 2D Filterbank Reconstruction
    using Ladder Structure

    x = fbrec_l(y0, y1, f, type1, type2, [extmod])

    Input:
    y0, y1: two input subband images
    f:  filter in the ladder network structure
    type1:  'q' or 'p' for selecting quincunx or parallelogram
    downsampling matrix
    type2:  second parameter for selecting the filterbank type
    If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
    ({1, 2, 0, 3} can also be used as equivalent)
    If type1 == 'p' then type2 is one of {0, 1, 2, 3}
    Those are specified in QPDEC and PPDEC
    extmod: [optional] extension mode (default is 'per')
    This refers to polyphase components.

    Output:
    x:  reconstructed image

    Note:       This is also called the lifting scheme

    See also:   FBDEC_L"""

    # Modulate f
    f[:, ::2] = -f[:, ::2]
    # Ladder network structure
    p1 = (-1 / sqrt(2)) * (y1 + sefilter2(y0, f, f, extmod))
    p0 = sqrt(2) * y0 + sefilter2(p1, f, f, extmod, array([[1], [1]]))

    # Polyphase reconstruction
    if str.lower(type1[0]) == 'q':
        # Quincunx polyphase reconstruction
        x = qprec(p0, p1, type2)
    elif str.lower(type1[0]) == 'p':
        # Parallelogram polyphase reconstruction
        x = pprec(p0, p1, type2)
    else:
        print 'Invalid argument type1'

    return x

# Retrive filters by names


def pfilters(fname):
    """ PFILTERS Generate filters for the laplacian pyramid

    Input:
    fname : Name of the filter, including the famous '9-7' filters.

    Output:
    h, g: 1D filters (lowpass for analysis and synthesis, respectively)
    for separable pyramid"""

    def filter97():
        h = array([[.037828455506995, -.023849465019380, -.11062440441842,
                   .37740285561265]])
        h = c_[h, .85269867900940, h[:, ::-1]]

        g = array([[-.064538882628938, -.040689417609558, .41809227322221]])
        g = c_[g, .78848561640566, g[:, ::-1]]

        return h, g

    def filterMaxFlat():
        M1 = 1 / sqrt(2)
        M2 = M1
        k1 = 1 - sqrt(2)
        k2 = M1
        k3 = k1
        h = array([[.25 * k2 * k3, .5 * k2, 1 + .5 * k2 * k3]]) * M1
        h = c_[h, h[:, size(h) - 2::-1]]

        g = array([[-.125 * k1 * k2 * k3, 0.25 * k1 * k2, -0.5 * k1 -
                  0.5 * k3 - 0.375 * k1 * k2 * k3, 1 + .5 * k1 * k2]]) * M2
        g = c_[g, g[:, size(g) - 2::-1]]
        # Normalize
        h = h * sqrt(2)
        g = g * sqrt(2)

        return h, g

    def filter53():
        h = array([[-1., 2., 6., 2., -1.]]) / (4 * sqrt(2))
        g = array([[1., 2., 1.]]) / (2 * sqrt(2))

        return h, g

    def filterBurt():
        h = array([[0.6, 0.25, -0.05]])
        h = sqrt(2) * c_[h[:, :0:-1], h]
        g = array([[17.0 / 28, 73.0 / 280, -3.0 / 56, -3.0 / 280]])
        g = sqrt(2) * c_[g[:, :0:-1], g]

        return h, g

    def filterPkva():
        # filters from the ladder structure
        # Allpass filter for the ladder structure network
        beta = ldfilter(fname)

        lf = size(beta)

        n = lf / 2.0

        if n != floor(n):
            print "The input allpass filter must be even length"

        # beta(z^2)
        beta2 = zeros((1, 2 * lf - 1))
        beta2[:, ::2] = beta

        # H(z)
        h = beta2.copy()
        h[:, 2 * n - 1] = h[:, 2 * n - 1] + 1
        h = h / 2.0

        # G(z)
        g = -signal.convolve(beta2, h)
        g[:, 4 * n - 2] = g[:, 4 * n - 2] + 1
        g[:, 1:-1:2] = -g[:, 1:-1:2]

        # Normalize
        h = h * sqrt(2)
        g = g * sqrt(2)

        return h, g

    def errhandler():
        print 'Invalid filter name'

    switch = {'9/7': filter97,
              '9-7': filter97,
              'maxflat': filterMaxFlat,
              '5/3': filter53,
              '5-3': filter53,
              'Burt': filterBurt,
              'burt': filterBurt,
              'pkva': filterPkva}

    return switch.get(fname, errhandler)()


def dfilters(fname, type):
    """ DFILTERS Generate directional 2D filters
    Input:
    fname:  Filter name.  Available 'fname' are:
    'haar': the Haar filters
    'vk':   McClellan transformed of the filter from the VK book
    'ko':   orthogonal filter in the Kovacevic's paper
    'kos':  smooth 'ko' filter
    'lax':  17 x 17 by Lu, Antoniou and Xu
    'sk':   9 x 9 by Shah and Kalker
    'cd':   7 and 9 McClellan transformed by Cohen and Daubechies
    'pkva': ladder filters by Phong et al.
    'oqf_362':  regular 3 x 6 filter
    'dvmlp':    regular linear phase biorthogonal filter with 3 dvm
    'sinc': ideal filter (*NO perfect recontruction*)
    'dmaxflat': diamond maxflat filters obtained from a three stage ladder

     type:  'd' or 'r' for decomposition or reconstruction filters

     Output:
    h0, h1: diamond filter pair (lowpass and highpass)

     To test those filters (for the PR condition for the FIR case),
     verify that:
     convolve(h0, modulate2(h1, 'b')) + convolve(modulate2(h0, 'b'), h1) = 2
     (replace + with - for even size filters)

     To test for orthogonal filter
     convolve(h, reverse2(h)) + modulate2(convolve(h, reverse2(h)), 'b') = 2
     """
    # The diamond-shaped filter pair
    def filterHaar():
        if str.lower(type[0]) == 'd':
            h0 = array([[1, 1]]) / sqrt(2)
            h1 = array([[-1, 1]]) / sqrt(2)
        else:
            h0 = array([[1, 1]]) / sqrt(2)
            h1 = array([[1, -1]]) / sqrt(2)
        return h0, h1

    def filterVk():
        if str.lower(type[0]) == 'd':
            h0 = array([[1, 2, 1]]) / 4.0
            h1 = array([[-1, -2, 6, -2, -1]]) / 4.0
        else:
            h0 = array([[-1, 2, 6, 2, -1]]) / 4.0
            h1 = array([[-1, 2, -1]]) / 4.0

        # McClellan transfrom to obtain 2D diamond filters
        t = array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0  # diamond kernel
        h0 = mctrans(h0, t)
        h1 = mctrans(h1, t)

        return h0, h1

    def filterKo():  # orthogonal filters in Kovacevic's thesis
        a0, a1, a2 = 2, 0.5, 1

        h0 = array([[0, -a1, -a0 * a1, 0],
                    [-a2, -a0 * a2, -a0, 1],
                    [0, a0 * a1 * a2, -a1 * a2, 0]])

        # h1 = qmf2(h0);
        h1 = array([[0, -a1 * a2, -a0 * a1 * a2, 0],
                    [1, a0, -a0 * a2, a2],
                    [0, -a0 * a1, a1, 0]])

        # Normalize filter sum and norm;
        norm = sqrt(2) / sum(h0)

        h0 = h0 * norm
        h1 = h1 * norm

        if str.lower(type[0]) == 'r':
            # Reverse filters for reconstruction
            h0 = h0[::-1, ::-1]
            h1 = h1[::-1, ::-1]

        return h0, h1

    def filterKos():  # Smooth orthogonal filters in Kovacevic's thesis
        a0, a1, a2 = -sqrt(3), -sqrt(3), 2 + sqrt(3)

        h0 = array([[0, -a1, -a0 * a1, 0],
                    [-a2, -a0 * a2, -a0, 1],
                    [0, a0 * a1 * a2, -a1 * a2, 0]])

        # h1 = qmf2(h0);

        h1 = array([[0, -a1 * a2, -a0 * a1 * a2, 0],
                    [1, a0, -a0 * a2, a2],
                    [0, -a0 * a1, a1, 0]])

        # Normalize filter sum and norm;

        norm = sqrt(2) / sum(h0)

        h0 = h0 * norm
        h1 = h1 * norm

        if str.lower(type[0]) == 'r':
            # Reverse filters for reconstruction
            h0 = h0[::-1, ::-1]
            h1 = h1[::-1, ::-1]

        return h0, h1

    def filterLax():    # by Lu, Antoniou and Xu
        h = array([[-1.2972901e-5, 1.2316237e-4, -7.5212207e-5, 6.3686104e-5,
                    9.4800610e-5, -7.5862919e-5, 2.9586164e-4, -1.8430337e-4],
                   [1.2355540e-4, -1.2780882e-4, -1.9663685e-5, -4.5956538e-5,
                    -6.5195193e-4, -2.4722942e-4, -2.1538331e-5, -7.0882131e-4],
                   [-7.5319075e-5, -1.9350810e-5, -7.1947086e-4, 1.2295412e-3,
                    5.7411214e-4, 4.4705422e-4, 1.9623554e-3, 3.3596717e-4],
                   [6.3400249e-5, -2.4947178e-4, 4.4905711e-4, -4.1053629e-3,
                    -2.8588307e-3, 4.3782726e-3, -3.1690509e-3, -3.4371484e-3],
                   [9.6404973e-5, -4.6116254e-5, 1.2371871e-3, -1.1675575e-2,
                    1.6173911e-2, -4.1197559e-3, 4.4911165e-3, 1.1635130e-2],
                   [-7.6955555e-5, -6.5618379e-4, 5.7752252e-4, 1.6211426e-2,
                    2.1310378e-2, -2.8712621e-3, -4.8422645e-2, -5.9246338e-3],
                   [2.9802986e-4, -2.1365364e-5, 1.9701350e-3, 4.5047673e-3,
                    -4.8489158e-2, -3.1809526e-3, -2.9406153e-2, 1.8993868e-1],
                   [-1.8556637e-4, -7.1279432e-4, 3.3839195e-4, 1.1662001e-2,
                    -5.9398223e-3, -3.4467920e-3, 1.9006499e-1, 5.7235228e-1]])

        h0 = sqrt(2) * vstack((hstack((h, h[:, len(h) - 2::-1])),
                               hstack((h[len(h) - 2::-1, :],
                                       h[len(h) - 2::-1, len(h) - 2::-1]))))
        h1 = modulate2(h0, 'b')

        return h0, h1

    def filterSk():  # by Shah and Kalker
        h = array([[0.621729, 0.161889, -0.0126949, -0.00542504, 0.00124838],
                  [0.161889, -0.0353769, -0.0162751, -0.00499353, 0],
                  [-0.0126949, -0.0162751, 0.00749029, 0, 0],
                  [-0.00542504, 0.00499353, 0, 0, 0],
                  [0.00124838, 0, 0, 0, 0]])

        h0 = sqrt(2) * vstack((hstack((h[len(h):0:-1, len(h):0:-1],
                                       h[len(h):0:-1, :])),
                               hstack((h[:, len(h):0:-1], h))))

        h1 = modulate2(h0, 'b')

        return h0, h1

    def filterDvmlp():
        q = sqrt(2)
        b = 0.02
        b1 = b * b
        h = array([[b / q, 0, -2 * q * b, 0, 3 * q * b, 0, -2 * q * b, 0, b / q],
                   [0, -1 / (16 * q), 0, 9 / (16 * q), 1 / q, 9 / (16 * q), 0, -1 / (16 * q), 0],
                   [b / q, 0, -2 * q * b, 0, 3 * q * b, 0, -2 * q * b, 0, b / q]])
        g0 = array([[-b1 / q, 0, 4 * b1 * q, 0, -14 * q * b1, 0, 28 * q * b1, 0, -35 * q * b1, 0,
                     28 * q * b1, 0, -14 * q * b1, 0, 4 * b1 * q, 0, -b1 / q],
                    [0, b / (8 * q), 0, -13 * b / (8 * q), b / q, 33 * b / (8 * q), -2 * q * b,
                     -21 * b / (8 * q), 3 * q * b, -21 * b / (8 * q), -2 * q * b, 33 * b / (8 * q),
                     b / q, -13 * b / (8 * q), 0, b / (8 * q), 0],
                    [-q * b1, 0, -1 / (256 * q) + 8 * q * b1, 0, 9 / (128 * q) - 28 * q * b1,
                     -1 / (q * 16), -63 / (256 * q) + 56 * q * b1, 9 / (16 * q),
                     87 / (64 * q) - 70 * q * b1, 9 / (16 * q), -63 / (256 * q) + 56 * q * b1,
                     -1 / (q * 16), 9 / (128 * q) - 28 * q * b1, 0, -1 / (256 * q) + 8 * q * b1,
                     0, -q * b1],
                    [0, b / (8 * q), 0, -13 * b / (8 * q), b / q, 33 * b / (8 * q), -2 * q * b,
                     -21 * b / (8 * q), 3 * q * b, -21 * b / (8 * q), -2 * q * b, 33 * b / (8 * q),
                     b / q, -13 * b / (8 * q), 0, b / (8 * q), 0],
                    [-b1 / q, 0, 4 * b1 * q, 0, -14 * q * b1, 0, 28 * q * b1, 0, -35 * q * b1,
                     0, 28 * q * b1, 0, -14 * q * b1, 0, 4 * b1 * q, 0, -b1 / q]])
        h1 = modulate2(g0, 'b')
        h0 = h.copy()
        if str.lower(type[0]) == 'r':
            h1 = modulate2(h, 'b')
            h0 = g0.copy()

        return h0, h1

    def filter79():  # by Cohen and Daubechies
        # 1D prototype filters: the '7-9' pair

        h0 = array([[0.026748757411, -0.016864118443, -0.078223266529,
                     0.266864118443, 0.602949018236, 0.266864118443,
                     -0.078223266529, -0.016864118443, 0.026748757411]])

        g0 = array([[-0.045635881557, -0.028771763114, 0.295635881557,
                    0.557543526229, 0.295635881557, -0.028771763114,
                    -0.045635881557]])

        if str.lower(type[0]) == 'd':
            h1 = modulate2(g0, 'c')
        else:
            h1 = modulate2(h0, 'c')
            h0 = g0.copy()

        # Use McClellan to obtain 2D filters
        t = array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0  # diamond kernel
        h0 = sqrt(2) * mctrans(h0, t)
        h1 = sqrt(2) * mctrans(h1, t)

        return h0, h1

    def filterPkva():
        # Filters from the ladder structure

        # Allpass filter for the ladder structure network

        beta = ldfilter(fname)

        # Analysis filters
        h0, h1 = ld2quin(beta)

        # Normalize norm
        h0 = sqrt(2) * h0
        h1 = sqrt(2) * h1

        # Synthesis filters
        if str.lower(type[0]) == 'r':
            f0 = modulate2(h1, 'b')
            f1 = modulate2(h0, 'b')
            h0 = f0.copy()
            h1 = f1.copy()

        return h0, h1

    def filterPkvaHalf4():  # Filters from the ladder structure
        # Allpass filter for the ladder structure network

        beta = ldfilterhalf(4)

        # Analysis filters

        h0, h1 = ld2quin(beta)

        # Normalize norm
        h0 = sqrt(2) * h0
        h1 = sqrt(2) * h1

        # Synthesis filters
        if str.lower(type[0]) == 'r':
            f0 = modulate2(h1, 'b')
            f1 = modulate2(h0, 'b')
            h0 = f0
            h1 = f1

        return h0, h1

    def filterPkvaHalf6():  # Filters from the ladder structure

        # Allpass filter for the ladder structure network
        beta = ldfilterhalf(6)

        # Analysis filters
        h0, h1 = ld2quin(beta)

        # Normalize norm
        h0 = sqrt(2) * h0
        h1 = sqrt(2) * h1

        # Synthesis filters
        if srtring.lower(type[0]) == 'r':
            f0 = modulate2(h1, 'b')
            f1 = modulate2(h0, 'b')
            h0 = f0
            h1 = f1

        return h0, h1

    def filterPkvaHalf8():  # Filters from the ladder structure
        # Allpass filter for the ladder structure network
        beta = ldfilterhalf(8)

        # Analysis filters
        h0, h1 = ld2quin(beta)

        # Normalize norm
        h0 = sqrt(2) * h0
        h1 = sqrt(2) * h1

        # Synthesis filters
        if str.lower(type[0]) == 'r':
            f0 = modulate2(h1, 'b')
            f1 = modulate2(h0, 'b')
            h0 = f0
            h1 = f1

        return h0, h1

    def filterSinc():   # The "sinc" case, NO Perfect Reconstruction

        # Ideal low and high pass filters
        flength = 30

        h0 = array([signal.filter_design.firwin(flength + 1, 0.5)])
        h1 = modulate2(h0, 'c')

        # Use McClellan to obtain 2D filters
        t = array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0  # diamond kernel
        h0 = sqrt(2) * mctrans(h0, t)
        h1 = sqrt(2) * mctrans(h1, t)

        return h0, h1

    def filterOqf():    # Some "home-made" filters!
        h0 = sqrt(2) / 64 * array([[sqrt(15), -3, 0],
                                   [0, 5, -sqrt(15)],
                                   [-2 * sqrt(15), 30, 0],
                                   [0, 30, 2 * sqrt(15)],
                                   [sqrt(15), 5, 0],
                                   [0, -3, -sqrt(15)]]).T

        h1 = -reverse2(modulate2(h0, 'b'))

        if str.lower(type[0]) == 'r':
            # Reverse filters for reconstruction
            h0 = h0[::-1, ::-1]
            h1 = h1[::-1, ::-1]

        return h0, h1

    def filterTest():      # Only for the shape, not for PR
        h0 = array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])
        h1 = array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        return h0, h1

    def filterTestDVM():  # Only for directional vanishing moment
        h0 = array([[1, 1], [1, 1]]) / sqrt(2)
        h1 = array([[-1, 1], [1, -1]]) / sqrt(2)

        return h0, h1

    def filterQmf():    # by Lu, Antoniou and Xu
        # ideal response
        # window
        m, n = 2, 2
        w = empty([5, 5])
        w1d = kaiser(4 * m + 1, 2.6)
        for n1 in xrange(-m, m + 1):
            for n2 in xrange(-n, n + 1):
                w[n1 + m, n2 + n] = w1d[2 * m + n1 + n2] * w1d[2 * m + n1 - n2]

        h = empty([5, 5])
        for n1 in xrange(-m, m + 1):
            for n2 in xrange(-n, n + 1):
                h[n1 + m, n2 + n] = .5 * \
                    sinc((n1 + n2) / 2.0) * .5 * sinc((n1 - n2) / 2.0)

        c = sum(h)
        h = sqrt(2) * h / c
        h0 = h * w
        h1 = modulate2(h0, 'b')

        return h0, h1
        #h0 = modulate2(h,'r');
        #h1 = modulate2(h,'b');

    def filterQmf2():   # by Lu, Antoniou and Xu
        # ideal response
        # window

        h = array([[-.001104, .002494, -0.001744, 0.004895,
                  -0.000048, -.000311],
                   [0.008918, -0.002844, -0.025197, -0.017135,
                  0.003905, -0.000081],
                   [-0.007587, -0.065904, 0.100431, -0.055878,
                  0.007023, 0.001504],
                   [0.001725, 0.184162, 0.632115, 0.099414,
                    -0.027006, -0.001110],
                   [-0.017935, -0.000491, 0.191397, -0.001787,
                    -0.010587, 0.002060],
                   [.001353, 0.005635, -0.001231, -0.009052,
                    -0.002668, 0.000596]])
        h0 = h / sum(h)
        h1 = modulate2(h0, 'b')

        return h0, h1

        #h0 = modulate2(h,'r');
        #h1 = modulate2(h,'b');

    def filterDmaxflat4():
        M1 = 1 / sqrt(2)
        M2 = M1
        k1 = 1 - sqrt(2)
        k3 = k1
        k2 = M1
        h = array([[.25 * k2 * k3, .5 * k2, 1 + .5 * k2 * k3]]) * M1
        h = c_[h, h[:, size(h) - 2::-1]]
        g = array([[-.125 * k1 * k2 * k3,
                    0.25 * k1 * k2,
                    (-0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3),
                    1 + .5 * k1 * k2]]) * M2
        g = c_[g, g[:, size(g) - 2::-1]]
        B = dmaxflat(4, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)
        h0 = sqrt(2) * h0 / sum(h0)
        g0 = sqrt(2) * g0 / sum(g0)

        h1 = modulate2(g0, 'b')

        if str.lower(type[0]) == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0.copy()

        return h0, h1

    def filterDmaxflat5():
        M1 = 1 / sqrt(2)
        M2 = M1
        k1 = 1 - sqrt(2)
        k3 = k1
        k2 = M1
        h = array([[.25 * k2 * k3, .5 * k2, 1 + .5 * k2 * k3]]) * M1
        h = c_[h, h[:, size(h) - 2::-1]]
        g = array([[-.125 * k1 * k2 * k3,
                    0.25 * k1 * k2,
                    (-0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3),
                    1 + .5 * k1 * k2]]) * M2
        g = c_[g, g[:, size(g) - 2::-1]]
        B = dmaxflat(5, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)
        h0 = sqrt(2) * h0 / sum(h0)
        g0 = sqrt(2) * g0 / sum(g0)

        h1 = modulate2(g0, 'b')
        if str.lower(type[0]) == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0.copy()
        return h0, h1

    def filterDmaxflat6():
        M1 = 1 / sqrt(2)
        M2 = M1
        k1 = 1 - sqrt(2)
        k3 = k1
        k2 = M1
        h = array([[.25 * k2 * k3, .5 * k2, 1 + .5 * k2 * k3]]) * M1
        h = c_[h, h[:, size(h) - 2::-1]]
        g = array([[-.125 * k1 * k2 * k3,
                    0.25 * k1 * k2,
                    (-0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3),
                    1 + .5 * k1 * k2]]) * M2
        g = c_[g, g[:, size(g) - 2::-1]]
        B = dmaxflat(6, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)
        h0 = sqrt(2) * h0 / sum(h0)
        g0 = sqrt(2) * g0 / sum(g0)

        h1 = modulate2(g0, 'b')

        if str.lower(type[0]) == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0.copy()

        return h0, h1

    def filterDmaxflat7():
        M1 = 1 / sqrt(2)
        M2 = M1
        k1 = 1 - sqrt(2)
        k3 = k1
        k2 = M1
        h = array([[.25 * k2 * k3, .5 * k2, 1 + .5 * k2 * k3]]) * M1
        h = c_[h, h[:, size(h) - 2::-1]]
        g = array([[-.125 * k1 * k2 * k3,
                    0.25 * k1 * k2,
                    (-0.5 * k1 - 0.5 * k3 - 0.375 * k1 * k2 * k3),
                    1 + .5 * k1 * k2]]) * M2
        g = c_[g, g[:, size(g) - 2::-1]]
        B = dmaxflat(7, 0)
        h0 = mctrans(h, B)
        g0 = mctrans(g, B)
        h0 = sqrt(2) * h0 / sum(h0)
        g0 = sqrt(2) * g0 / sum(g0)

        h1 = modulate2(g0, 'b')

        if str.lower(type[0]) == 'r':
            h1 = modulate2(h0, 'b')
            h0 = g0.copy()

        return h0, h1

    def errhandler():
        print 'Unrecognized ladder structure filter name'

    switch = {'haar': filterHaar,
              'vk': filterVk,
              'ko': filterKo,
              'kos': filterKos,
              'lax': filterLax,
              'sk': filterSk,
              'cd': filter79,
              '7-9': filter79,
              'pkva': filterPkva,
              'pkva-half4': filterPkvaHalf4,
              'pkva-half6': filterPkvaHalf6,
              'pkva-half8': filterPkvaHalf8,
              'oqf_362': filterOqf,
              'test': filterTest,
              'dvmlp': filterDvmlp,
              'testDVM': filterTestDVM,
              'qmf': filterQmf,
              'qmf2': filterQmf2,
              'sinc': filterSinc,
              'dmaxflat4': filterDmaxflat4,
              'dmaxflat5': filterDmaxflat5,
              'dmaxflat6': filterDmaxflat6,
              'dmaxflat7': filterDmaxflat7}

    return switch.get(fname, errhandler)()


def ldfilter(fname):
    """LDFILTER Generate filter for the ladder structure network
    f = ldfilter(fname)

    Input: fname:  Available 'fname' are:
    'pkvaN': length N filter from Phoong, Kim, Vaidyanathan and Ansari"""

    def pkva12():
        v = array([[0.6300, -0.1930, 0.0972, -0.0526, 0.0272, -0.0144]])
        return v

    def pkva8():
        v = array([[0.6302, -0.1924, 0.0930, -0.0403]])
        return v

    def pkva6():
        v = array([[0.6261, -0.1794, 0.0688]])
        return v

    def errhandler():
        print 'Unrecognized ladder structure filter name'
    switch = {'pkva': pkva12,
              'pkva12': pkva12,
              'pkva8': pkva8,
              'pkva6': pkva6}
    v = switch.get(fname, errhandler)()
    # Symmetric impulse response
    f = c_[v[:, ::-1], v]
    return f


def dmaxflat(N, d):
    """returns 2-D diamond maxflat filters of order 'N'
    the filters are nonseparable and 'd' is the (0,0) coefficient,
    being 1 or 0 depending on use
    """
    if (N > 7 or N < 1):
        print 'N must be in {1,2,3,4,5,6,7}'

    def dmaxflat1():
        h = array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0
        h[2, 2] = d
        return h

    def dmaxflat2():
        h = array([[0, -1, 0], [-1, 0, 10], [0, 10, 0]])
        h = c_[h, fliplr(h[:, :-1])]
        h = r_[h, flipud(h[:-1, :])] / 32.0
        h[3, 3] = d
        return h

    def dmaxflat3():
        h = array([[0, 3, 0, 2],
                   [3, 0, -27, 0],
                   [0, -27, 0, 174],
                   [2, 0, 174, 0]])
        h = c_[h, fliplr(h[:, :-1])]
        h = r_[h, flipud(h[:-1, :])] / 512.0
        h[4, 4] = d
        return h

    def dmaxflat4():
        h = array([[0, -5, 0, -3, 0],
                   [-5, 0, 52, 0, 34],
                   [0, 52, 0, -276, 0],
                   [-3, 0, -276, 0, 1454],
                   [0, 34, 0, 1454, 0]]) / 2.0**12
        h = c_[h, fliplr(h[:, :-1])]
        h = r_[h, flipud(h[:-1, :])]
        h[5, 5] = d
        return h

    def dmaxflat5():
        h = array([[0, 35, 0, 20, 0, 18],
                   [35, 0, -425, 0, -250, 0],
                   [0, -425, 0, 2500, 0, 1610],
                   [20, 0, 2500, 0, -10200, 0],
                   [0, -250, 0, -10200, 0, 47780],
                   [18, 0, 1610, 0, 47780, 0]]) / 2.0**17
        h = c_[h, fliplr(h[:, :-1])]
        h = r_[h, flipud(h[:-1, :])]
        h[6, 6] = d
        return h

    def dmaxflat6():
        h = array([[0, -63, 0, -35, 0, -30, 0],
                   [-63, 0, 882, 0, 495, 0, 444],
                   [0, 882, 0, -5910, 0, -3420, 0],
                   [-35, 0, -5910, 0, 25875, 0, 16460],
                   [0, 495, 0, 25875, 0, -89730, 0],
                   [-30, 0, -3420, 0, -89730, 0, 389112],
                   [0, 44, 0, 16460, 0, 389112, 0]]) / 2.0**20
        h = c_[h, fliplr(h[:, :-1])]
        h = r_[h, flipud(h[:-1, :])]
        h[7, 7] = d
        return h

    def dmaxflat7():
        h = array([[0, 231, 0, 126, 0, 105, 0, 100],
                   [231, 0, -3675, 0, -2009, 0, -1715, 0],
                   [0, -3675, 0, 27930, 0, 15435, 0, 13804],
                   [126, 0, 27930, 0, -136514, 0, -77910, 0],
                   [0, -2009, 0, -136514, 0, 495145, 0, 311780],
                   [105, 0, 15435, 0, 495145, 0, -1535709, 0],
                   [0, -1715, 0, -77910, 0, -1535709, 0, 6305740],
                   [100, 0, 13804, 0, 311780, 0, 6305740, 0]]) / 2.0**24
        h = c_[h, fliplr(h[:, :-1])]
        h = r_[h, flipud(h[:-1, :])]
        h[8, 8] = d
        return h

    def errhandler():
        print 'Invalid argument type'
    switch = {1: dmaxflat1,
              2: dmaxflat2,
              3: dmaxflat3,
              4: dmaxflat4,
              5: dmaxflat5,
              6: dmaxflat6,
              7: dmaxflat7}

    return switch.get(N, errhandler)()

# Multidimensional filtering (used in building block filter banks)


def sefilter2(x, f1, f2, extmod='per', shift=array([[0], [0]])):
    """SEFILTER2   2D seperable filtering with extension handling
    y = sefilter2(x, f1, f2, [extmod], [shift])

    Input:
    x:      input image
    f1, f2: 1-D filters in each dimension that make up a 2D seperable filter
    extmod: [optional] extension mode (default is 'per')
    shift:  [optional] specify the window over which the
    convolution occurs. By default shift = [0; 0].

    Output:
    y:      filtered image of the same size as the input image:
    Y(z1,z2) = X(z1,z2)*F1(z1)*F2(z2)*z1^shift(1)*z2^shift(2)

    Note:
    The origin of the filter f is assumed to be floor(size(f)/2) + 1.
    Amount of shift should be no more than floor((size(f)-1)/2).
   The output image has the same size with the input image.

   See also: EXTEND2, EFILTER2"""

    # Make sure filter in a row vector
    f1 = f1.flatten(1)[:, newaxis].T
    f2 = f2.flatten(1)[:, newaxis].T

    # Periodized extension
    lf1 = (size(f1) - 1) / 2.0
    lf2 = (size(f1) - 1) / 2.0

    y = extend2(x, floor(lf1) + shift[0, 0], ceil(lf1) - shift[0, 0],
                floor(lf2) + shift[1, 0], ceil(lf2) - shift[1, 0], extmod)
    # pdb.set_trace()
    # Seperable filter
    y = signal.convolve(y, f1, 'valid')
    y = signal.convolve(y, f2.T, 'valid')
    return y


def efilter2(x, f, extmod='per', shift=array([[0], [0]])):
    """EFILTER2   2D Filtering with edge handling (via extension)

    y = efilter2(x, f, [extmod], [shift])

    Input:
    x:  input image
    f:  2D filter
    extmod: [optional] extension mode (default is 'per')
    shift:  [optional] specify the window over which the
    convolution occurs. By default shift = [0; 0].

    Output:
    y:  filtered image that has:
    Y(z1,z2) = X(z1,z2)*F(z1,z2)*z1^shift(1)*z2^shift(2)

    Note:
    The origin of filter f is assumed to be floor(size(f)/2) + 1.
    Amount of shift should be no more than floor((size(f)-1)/2).
    The output image has the same size with the input image.

    See also:   EXTEND2, SEFILTER2"""

    # Periodized extension
    sf = (array(f.shape) - 1) / 2.0

    xext = extend2(x, floor(sf[0]) +
                   shift[0, 0], ceil(sf[0]) -
                   shift[0, 0], floor(sf[1]) +
                   shift[1, 0], ceil(sf[1]) -
                   shift[1, 0], extmod)

    # Convolution and keep the central part that has the size as the input
    y = signal.convolve(xext, f, 'valid')

    return y


def extend2(x, ru, rd, cl, cr, extmod):
    """ EXTEND2   2D extension
    y = extend2(x, ru, rd, cl, cr, extmod)

    Input:
    x:  input image
    ru, rd: amount of extension, up and down, for rows
    cl, cr: amount of extension, left and rigth, for column
    extmod: extension mode.  The valid modes are:
    'per':      periodized extension (both direction)
    'qper_row': quincunx periodized extension in row
    'qper_col': quincunx periodized extension in column

    Output:
    y:  extended image

    Note:
    Extension modes 'qper_row' and 'qper_col' are used multilevel
    quincunx filter banks, assuming the original image is periodic in
    both directions.  For example:
    [y0, y1] = fbdec(x, h0, h1, 'q', '1r', 'per');
    [y00, y01] = fbdec(y0, h0, h1, 'q', '2c', 'qper_col');
    [y10, y11] = fbdec(y1, h0, h1, 'q', '2c', 'qper_col');

    See also:   FBDEC"""

    rx, cx = array(x.shape)

    def extmodPer():
        I = getPerIndices(rx, ru, rd)
        y = x[I, :]

        I = getPerIndices(cx, cl, cr)
        y = y[:, I]

        return y

    def extmodQper_row():
        rx2 = round(rx / 2.0)
        y = c_[r_[x[rx2:rx, cx - cl:cx], x[0:rx2, cx - cl:cx]],
               x, r_[x[rx2:rx, 0:cr], x[0:rx2, 0:cr]]]
        I = getPerIndices(rx, ru, rd)
        y = y[I, :]

        return y

    def extmodQper_col():
        cx2 = round(cx / 2.0)
        y = r_[c_[x[rx - ru:rx, cx2:cx], x[rx - ru:rx, 0:cx2]],
               x, c_[x[0:rd, cx2:cx], x[0:rd, 0:cx2]]]

        I = getPerIndices(cx, cl, cr)
        y = y[:, I]

        return y

    def errhandler():
        print 'Invalid input for EXTMOD'

    switch = {'per': extmodPer,
              'qper_row': extmodQper_row,
              'qper_col': extmodQper_col}

    return switch.get(extmod, errhandler)()

#----------------------------------------------------------------------------#
# Internal Function(s)
#----------------------------------------------------------------------------#


def getPerIndices(lx, lb, le):
    I = r_[arange(lx - lb + 1, lx + 1), arange(1, lx + 1), arange(1, le + 1)]
    if (lx < lb) or (lx < le):
        I = mod(I, lx)
        I[I == 0] = lx
    I = I - 1

    return I.astype(int)

# Multidimesional Sampling (used in building block filter banks)


def pdown(x, type, phase=0):
    """ PDOWN   Parallelogram Downsampling
        y = pdown(x, type, [phase])
     Input:
        x:  input image
        type:   one of {0, 1, 2, 3} for selecting sampling matrices:
                P0 = [2, 0; 1, 1]
                P1 = [2, 0; -1, 1]
                P2 = [1, 1; 0, 2]
                P3 = [1, -1; 0, 2]
        phase:  [optional] 0 or 1 for keeping the zero- or one-polyphase
            component, (default is 0)
     Output:
        y:  parallelogram downsampled image
     Note:
        These sampling matrices appear in the directional filterbank:
            P0 = R0 * Q0
            P1 = R1 * Q1
            P2 = R2 * Q1
            P3 = R3 * Q0
        where R's are resampling matrices and Q's are quincunx matrices
     See also:  PPDEC"""
    # Parallelogram polyphase decomposition by simplifying sampling matrices
    # using the Smith decomposition of the quincunx matrices
    def type0():  # P0 = R0 * Q0 = D0 * R2
        if phase == 0:
            y = resamp(x[::2], 2)
        else:
            y = resamp(x[1::2, r_[1:len(x), 0]], 2)

        return y

    def type1():  # P1 = R1 * Q1 = D0 * R3
        if phase == 0:
            y = resamp(x[::2], 3)
        else:
            y = resamp(x[1::2], 3)

        return y

    def type2():  # P2 = R2 * Q1 = D1 * R0
        if phase == 0:
            y = resamp(x[:, ::2], 0)
        else:
            y = resamp(x[r_[1:len(x), 0], 1::2], 0)

        return y

    def type3():  # P3 = R3 * Q0 = D1 * R1
        if phase == 0:
            y = resamp(x[:, ::2], 1)
        else:
            y = resamp(x[:, 1::2], 1)

        return y

    def errhandler():
        print 'Invalid argument type'

    switch = {0: type0,
              1: type1,
              2: type2,
              3: type3}

    return switch.get(type, errhandler)()


def pup(x, type, phase=0):
    """ PUP   Parallelogram Upsampling

        y = pup(x, type, [phase])

     Input:
        x:  input image
        type:   one of {0, 1, 2, 3} for selecting sampling matrices:
                P0 = [2, 0; 1, 1]
                P1 = [2, 0; -1, 1]
                P2 = [1, 1; 0, 2]
                P3 = [1, -1; 0, 2]
        phase:  [optional] 0 or 1 to specify the phase of the input image as
            zero- or one-polyphase  component, (default is 0)

     Output:
        y:  parallelogram upsampled image

     Note:
        These sampling matrices appear in the directional filterbank:
            P0 = R0 * Q0
            P1 = R1 * Q1
            P2 = R2 * Q1
            P3 = R3 * Q0
        where R's are resampling matrices and Q's are quincunx matrices

     See also:  PPDEC"""

    # Parallelogram polyphase decomposition by simplifying sampling matrices
    # using the Smith decomposition of the quincunx matrices
    #
    # Note that R0 * R1 = R2 * R3 = I so for example,
    # upsample by R0 is the same with down sample by R1.
    # Also the order of upsampling operations is in the reserved order
    # with the one of matrix multiplication.

    m, n = x.shape

    def type0():  # P0 = R0 * Q0 = D0 * R2
        y = zeros((2 * m, n))
        if phase == 0:
            y[::2] = resamp(x, 3)
        else:
            y[1::2, r_[1:len(y), 0]] = resamp(x, 3)

        return y

    def type1():  # P1 = R1 * Q1 = D0 * R3
        y = zeros((2 * m, n))
        if phase == 0:
            y[::2] = resamp(x, 2)
        else:
            y[1::2] = resamp(x, 2)

        return y

    def type2():  # P2 = R2 * Q1 = D1 * R0
        y = zeros((m, 2 * n))
        if phase == 0:
            y[:, ::2] = resamp(x, 1)
        else:
            y[r_[1:len(y), 0], 1::2] = resamp(x, 1)

        return y

    def type3():  # P3 = R3 * Q0 = D1 * R1
        y = zeros((m, 2 * n))
        if phase == 0:
            y[:, ::2] = resamp(x, 0)
        else:
            y[:, 1::2] = resamp(x, 0)

        return y

    def errhandler():
        print 'Invalid argument type'

    switch = {0: type0,
              1: type1,
              2: type2,
              3: type3}

    return switch.get(type, errhandler)()


def qdown(x, type='1r', extmod='per', phase=0):
    """ QDOWN   Quincunx Downsampling

        y = qdown(x, [type], [extmod], [phase])

     Input:
        x:  input image
        type:   [optional] one of {'1r', '1c', '2r', '2c'} (default is '1r')
            '1' or '2' for selecting the quincunx matrices:
                Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
            'r' or 'c' for suppresing row or column
        phase:  [optional] 0 or 1 for keeping the zero- or one-polyphase
            component, (default is 0)

     Output:
        y:  qunincunx downsampled image

     See also:  QPDEC"""

    """ Quincunx downsampling using the Smith decomposition:
        Q1 = R1 * [2, 0; 0, 1] * R2
           = R2 * [1, 0; 0, 2] * R1
     and,
        Q2 = R0 * [2, 0; 0, 1] * R3
           = R3 * [1, 0; 0, 2] * R0

    See RESAMP for the definition of those resampling matrices"""

    def type1r():
        z = resamp(x, 1)
    if phase == 0:
        y = resamp(z[::2], 2)
    else:
        y = resamp(z[1::2, r_[1:len(z), 0]], 2)
        return y

    def type1c():
        z = resamp(x, 2)
        if phase == 0:
            y = resamp(z[:, ::2], 1)
        else:
            y = resamp(z[:, 1::2], 1)
        return y

    def type2r():
        z = resamp(x, 0)
        if phase == 0:
            y = resamp(z[::2, :], 3)
        else:
            y = resamp(z[1::2, :], 3)
        return y

    def type2c():
        z = resamp(x, 3)
        if phase == 0:
            y = resamp(z[:, ::2], 0)
        else:
            y = resamp(z[r_[1:len(z), 0], 1::2], 0)
        return y

    def errhandler():
        print 'Invalid argument type'

    switch = {'1r': type1r,
              '1c': type1c,
              '2r': type2r,
              '2c': type2c}
    return switch.get(type, errhandler)()


def qup(x, type='1r', phase=0):
    """ QUP   Quincunx Upsampling

        y = qup(x, [type], [phase])

     Input:
        x:  input image
        type:   [optional] one of {'1r', '1c', '2r', '2c'} (default is '1r')
            '1' or '2' for selecting the quincunx matrices:
                Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
            'r' or 'c' for extending row or column
        phase:  [optional] 0 or 1 to specify the phase of the input image as
            zero- or one-polyphase component, (default is 0)

     Output:
        y:  qunincunx upsampled image

     See also:  QDOWN"""

    """ Quincunx downsampling using the Smith decomposition:

        Q1 = R2 * [2, 0; 0, 1] * R3
           = R3 * [1, 0; 0, 2] * R2
     and,
        Q2 = R1 * [2, 0; 0, 1] * R4
           = R4 * [1, 0; 0, 2] * R1

     See RESAMP for the definition of those resampling matrices

     Note that R0 * R1 = R2 * R3 = I so for example,
     upsample by R0 is the same with down sample by R1.
     Also the order of upsampling operations is in the reserved order
     with the one of matrix multiplication."""

    m, n = x.shape

    def type1r():
        z = zeros((2 * m, n))

        if phase == 0:
            z[::2] = resamp(x, 3)
        else:
            z[1::2, r_[1:len(z), 0]] = resamp(x, 3)
        y = resamp(z, 0)

        return y

    def type1c():
        z = zeros((m, 2 * n))
        if phase == 0:
            z[:, ::2] = resamp(x, 0)
        else:
            z[:, 1::2] = resamp(x, 0)
        y = resamp(z, 3)

        return y

    def type2r():
        z = zeros((2 * m, n))
        if phase == 0:
            z[::2, :] = resamp(x, 2)
        else:
            z[1::2, :] = resamp(x, 2)
        y = resamp(z, 1)

        return y

    def type2c():
        z = zeros((m, 2 * n))
        if phase == 0:
            z[:, ::2] = resamp(x, 1)
        else:
            z[r_[1:len(z), 0], 1::2] = resamp(x, 1)
        y = resamp(z, 2)

        return y

    def errhandler():
        print 'Invalid argument type'

    switch = {'1r': type1r,
              '1c': type1c,
              '2r': type2r,
              '2c': type2c}

    return switch.get(type, errhandler)()


def qupz(x, type=1):
    """ QUPZ   Quincunx Upsampling (with zero-pad and matrix extending)
        y = qup(x, [type])
        Input:
    x:  input image
    type:   [optional] 1 or 2 for selecting the quincunx matrices:
            Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
        Output:
    y:  qunincunx upsampled image

       This resampling operation does NOT involve periodicity, thus it
       zero-pad and extend the matrix"""

    """ Quincunx downsampling using the Smith decomposition:
    Q1 = R1 * [2, 0; 0, 1] * R2
        and,
    Q2 = R0 * [2, 0; 0, 1] * R3

        See RESAMP for the definition of those resampling matrices

     Note that R0 * R1 = R2 * R3 = I so for example,
        upsample by R1 is the same with down sample by R2.
        Also the order of upsampling operations is in the reserved order
        with the one of matrix multiplication."""

    def type1():
        x1 = resampz(x, 3)
        m, n = x1.shape
        x2 = zeros((2 * m - 1, n))
        x2[::2] = x1.copy()
        y = resampz(x2, 0)

        return y

    def type2():
        x1 = resampz(x, 2)
        m, n = x1.shape
        x2 = zeros((2 * m - 1, n))
        x2[::2] = x1.copy()
        y = resampz(x2, 1)

        return y

    def errhandler():
        print 'Invalid argument type'

    switch = {1: type1,
              2: type2}

    return switch.get(type, errhandler)()


def dup(x, step, phase=array([0, 0])):
    """ DUP   Diagonal Upsampling

    y = dup(x, step, [phase])

    Input:
    x:  input image
    step:   upsampling factors for each dimension which should be a
    2-vector
    phase:  [optional] to specify the phase of the input image which
    should be less than step, (default is [0, 0])
    If phase == 'minimum', a minimum size of upsampled image
    is returned

    Output:
    y:  diagonal upsampled image

    See also:   DDOWN"""

    sx = array(x.shape)

    if phase[0] == 'm' or phase[0] == 'M':
        y = zeros((sx - 1) * step + 1)
        y[0::step[0], 0::step[0]] = x.copy()
    else:
        y = zeros(sx * step)
        y[phase[0]::step[0], phase[1]::step[1]] = x.copy()

    return y


def resamp(x, type, shift=1, extmod='per'):
    """ RESAMP   Resampling in 2D filterbank

    y = resamp(x, type, [shift, extmod])

        Input:
    x:  input image
        type: one of {0,1,2,3} (see note)

    shift:  [optional] amount of shift (default is 1)
        extmod: [optional] extension mode (default is 'per').
        Other options are:

        Output:
    y:  resampled image.

        Note:
    The resampling matrices are:
        R0 = [1, 1;  0, 1];
        R1 = [1, -1; 0, 1];
        R2 = [1, 0;  1, 1];
        R3 = [1, 0; -1, 1];

    For type 0 and type 1, the input image is extended (for example
    periodically) along the vertical direction;
    while for type 2 and type 3 the image is extended along the
    horizontal direction.

    Calling resamp(x, type, n) which n is positive integer is equivalent
    to repeatly calling resamp(x, type) n times.

    Input shift can be negative so that resamp(x, 0, -1) is the same
    with resamp(x, 1, 1)"""
    def type01():
        y = resampc(x, type, shift, extmod)
        return y

    def type23():
        y = resampc(x.T, type - 2, shift, extmod).T
        return y

    def errhandler():
        print 'The second input (type) must be one of {0, 1, 2, 3}'

    switch = {0: type01,
              1: type01,
              2: type23,
              3: type23}

    return switch.get(type, errhandler)()


def resampz(x, type, shift=1):
    """ RESAMPZ   Resampling of matrix
        y = resampz(x, type, [shift])

        Input:
        x:      input matrix
        type:   one of {0, 1, 2, 3} (see note)
        shift:  [optional] amount of shift (default is 1)

        Output:
        y:      resampled matrix

        Note:
    The resampling matrices are:
        R1 = [1, 1;  0, 1];
        R2 = [1, -1; 0, 1];
        R3 = [1, 0;  1, 1];
        R4 = [1, 0; -1, 1];

    This resampling program does NOT involve periodicity, thus it
    zero-pad and extend the matrix."""
    sx = array(x.shape)

    def type01():
        y = zeros([sx[0] + abs(shift * (sx[1] - 1)), sx[1]])
        if type == 0:
            shift1 = arange(sx[1]) * (-shift)
        else:
            shift1 = arange(sx[1]) * shift

        # Adjust to non-negative shift if needed
        if shift1[-1] < 0:
            shift1 = shift1 - shift1[-1]

        for n in xrange(sx[1]):
            y[shift1[n] + arange(sx[0]), n] = x[:, n].copy()

        # Finally, delete zero rows if needed
        start = 0
        finish = array(y.shape[0])
        while linalg.norm(y[start, :]) == 0:
            start = start + 1

        while linalg.norm(y[finish - 1, :]) == 0:
            finish = finish - 1

        y = y[start:finish, :]

        return y

    def type23():
        y = zeros([sx[0], sx[1] + abs(shift * (sx[0] - 1))])
        if type == 2:
            shift2 = arange(sx[0]) * (-shift)
        else:
            shift2 = arange(sx[0]) * shift

        # Adjust to non-negative shift if needed
        if shift2[-1] < 0:
            shift2 = shift2 - shift2[-1]

        for m in xrange(sx[0]):
            y[m, shift2[m] + arange(sx[1])] = x[m, :].copy()

        # Finally, delete zero columns if needed
        start = 0
        finish = array(y.shape[1])

        while linalg.norm(y[:, start]) == 0:
            start = start + 1
        while linalg.norm(y[:, finish - 1]) == 0:
            finish = finish - 1

        y = y[:, start:finish]

        return y

    def errhandler():
        print 'The second input (type) must be one of {0, 1, 2, 3}'

    switch = {0: type01,
              1: type01,
              2: type23,
              3: type23}

    return switch.get(type, errhandler)()


def resampc(x, type, shift=1, extmod='per'):
    """ RESAMPC Resampling along the column

    y = resampc(x, type, shift, extmod)

    Input:
    x:  image that is extendable along the column direction
    type:   either 0 or 1 (0 for shuffering down and 1 for up)
    shift:  amount of shifts (typically 1)
    extmod: extension mode:
    'per'   periodic
    'ref1'  reflect about the edge pixels
    'ref2'  reflect, doubling the edge pixels

    Output:
    y:  resampled image with:
    R1 = [1, shift; 0, 1] or R2 = [1, -shift; 0, 1]"""

    if type != 0 and type != 1:
        print 'The second input (type) must be either 0 or 1'
        return

    if is_string_like(extmod) != 1:
        print 'EXTMOD arg must be a string'
        return

    m, n = x.shape
    y = zeros(x.shape)
    s = shift

    if extmod == 'per':
        code = """
        int i, j, k;
        for(j = 0; j < n; j++) {
            /* Circular shift in each column */
            if(type == 0)
                k = (s * j) % m;
            else
                k = (-s * j) % m;

            /* Convert to non-negative mod if needed */

            if(k < 0)
                k += m;

            for(i = 0; i < m; i++) {
                if (k >= m)
                    k -= m;
                y(i, j) = x(k, j);
                k++
            }
        }
      """
    else:
        print 'Invalid exrmod'
    # call weave
    weave.inline(code, ['m', 'n', 'x', 'y', 's', 'type'],
                 type_converters=converters.blitz, compiler='gcc')

    return y

# Polyphase decomposition (used in the ladder structure implementation)


def qpdec(x, type='1r'):
    """ QPDEC   Quincunx Polyphase Decomposition

        [p0, p1] = qpdec(x, [type])

     Input:
        x:  input image
        type:   [optional] one of {'1r', '1c', '2r', '2c'} default is '1r'
            '1' and '2' for selecting the quincunx matrices:
                Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
            'r' and 'c' for suppresing row or column

     Output:
        p0, p1: two qunincunx polyphase components of the image"""

    """ Quincunx downsampling using the Smith decomposition:
        Q1 = R1 * D0 * R2
           = R2 * D1 * R1
     and,
        Q2 = R0 * D0 * R3
           = R3 * D1 * R0

     where D0 = [2, 0; 0, 1] and D1 = [1, 0; 0, 2].
     See RESAMP for the definition of the resampling matrices R's"""

    def type1r():  # Q1 = R1 * D0 * R2
        y = resamp(x, 1)
        p0 = resamp(y[::2], 2)
        # inv(R2) * [0; 1] = [1; 1]
        p1 = resamp(y[1::2, r_[1:len(y), 0]], 2)

        return p0, p1

    def type1c():  # Q1 = R2 * D1 * R1
        y = resamp(x, 2)
        p0 = resamp(y[:, ::2], 1)
        # inv(R3) * [0; 1] = [0; 1]
        p1 = resamp(y[:, 1::2], 1)

        return p0, p1

    def type2r():  # Q2 = R0 * D0 * R3
        y = resamp(x, 0)
        p0 = resamp(y[::2], 3)
        # inv(R1) * [1; 0] = [1; 0]
        p1 = resamp(y[1::2], 3)

        return p0, p1

    def type2c():  # Q2 = R3 * D1 * R0
        y = resamp(x, 3)
        p0 = resamp(y[:, ::2], 0)
        # inv(R4) * [1; 0] = [1; 1]
        p1 = resamp(y[r_[1:len(y), 0], 1::2], 0)
        return p0, p1

    def errhandler():
        print 'Invalid argument type'

    switch = {'1r': type1r,
              '1c': type1c,
              '2r': type2c,
              '2c': type2c}

    return switch.get(type, errhandler)()


def qprec(p0, p1, type='1r'):
    """ QPREC   Quincunx Polyphase Reconstruction

        x = qprec(p0, p1, [type])

     Input:
        p0, p1: two qunincunx polyphase components of the image
        type:   [optional] one of {'1r', '1c', '2r', '2c'}, default is '1r'
            '1' and '2' for selecting the quincunx matrices:
                Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
            'r' and 'c' for suppresing row or column

     Output:
        x:  reconstructed image

     Note:
        Note that R1 * R2 = R3 * R4 = I so for example,
        upsample by R1 is the same with down sample by R2

     See also:  QPDEC"""

    """ Quincunx downsampling using the Smith decomposition:

           Q1 = R1 * D0 * R2
              = R2 * D1 * R1
     and,
           Q2 = R0 * D0 * R3
              = R3 * D1 * R0

     where D0 = [2, 0; 0, 1] and D1 = [1, 0; 0, 2].
     See RESAMP for the definition of the resampling matrices R's"""

    m, n = p0.shape

    def type1r():  # Q1 = R2 * D1 * R3
        y = zeros((2 * m, n))
        y[::2, :] = resamp(p0, 3)
        y[1::2, r_[1:len(y), 0]] = resamp(p1, 3)
        x = resamp(y, 0)

        return x

    def type1c():  # Q1 = R3 * D2 * R2
        y = zeros((m, 2 * n))
        y[:, ::2] = resamp(p0, 0)
        y[:, 1::2] = resamp(p1, 0)
        x = resamp(y, 3)

        return x

    def type2r():  # Q2 = R1 * D1 * R4
        y = zeros((2 * m, n))
        y[::2, :] = resamp(p0, 2)
        y[1::2, :] = resamp(p1, 2)
        x = resamp(y, 1)
        return x

    def type2c():  # Q2 = R4 * D2 * R1
        y = zeros((m, 2 * n))
        y[:, ::2] = resamp(p0, 1)
        y[r_[1:len(y), 0], 1::2] = resamp(p1, 1)
        x = resamp(y, 2)
        return x

    def errhandler():
        print 'Invalid argument type'

    switch = {'1r': type1r,
              '1c': type1c,
              '2r': type2c,
              '2c': type2c}

    return switch.get(type, errhandler)()


def ppdec(x, type):
    """ PPDEC   Parallelogram Polyphase Decomposition

        [p0, p1] = ppdec(x, type)

     Input:
        x:  input image
        type:   one of {1, 2, 3, 4} for selecting sampling matrices:
                P0 = [2, 0; 1, 1]
                P1 = [2, 0; -1, 1]
                P2 = [1, 1; 0, 2]
                P3 = [1, -1; 0, 2]

     Output:
        p0, p1: two parallelogram polyphase components of the image

     Note:
        These sampling matrices appear in the directional filterbank:
            P0 = R0 * Q1
            P1 = R1 * Q2
            P2 = R2 * Q2
            P3 = R3 * Q1
        where R's are resampling matrices and Q's are quincunx matrices

     See also:  QPDEC"""

    # Parallelogram polyphase decomposition by simplifying sampling matrices
    # using the Smith decomposition of the quincunx matrices

    def type0():  # P0 = R0 * Q1 = D0 * R2
        p0 = resamp(x[::2, :], 2)
        # R0 * [0; 1] = [1; 1]
        p1 = resamp(x[1::2, r_[1:len(x), 0]], 2)

        return p0, p1

    def type1():  # P1 = R1 * Q2 = D0 * R3
        p0 = resamp(x[::2, :], 3)
        # R1 * [1; 0] = [1; 0]
        p1 = resamp(x[1::2, :], 3)

        return p0, p1

    def type2():  # P2 = R2 * Q2 = D1 * R0
        p0 = resamp(x[:, ::2], 0)

        # R2 * [1; 0] = [1; 1]
        p1 = resamp(x[r_[1:len(x), 0], 1::2], 0)

        return p0, p1

    def type3():  # P3 = R3 * Q1 = D1 * R1
        p0 = resamp(x[:, ::2], 1)

        # R3 * [0; 1] = [0; 1]
        p1 = resamp(x[:, 1::2], 1)

        return p0, p1

    def errhandler():
        print 'Invalid argument type'

    switch = {0: type0,
              1: type1,
              2: type2,
              3: type3}

    return switch.get(type, errhandler)()


def pprec(p0, p1, type):
    """ PPREC   Parallelogram Polyphase Reconstruction

        x = pprec(p0, p1, type)

     Input:
        p0, p1: two parallelogram polyphase components of the image
        type:   one of {0, 1, 2, 3} for selecting sampling matrices:
                P0 = [2, 0; 1, 1]
                P1 = [2, 0; -1, 1]
                P2 = [1, 1; 0, 2]
                P3 = [1, -1; 0, 2]

     Output:
        x:  reconstructed image

     Note:
        These sampling matrices appear in the directional filterbank:
            P0 = R0 * Q1
            P1 = R1 * Q2
            P2 = R2 * Q2
            P3 = R3 * Q1
        where R's are resampling matrices and Q's are quincunx matrices

        Also note that R0 * R1 = R2 * R3 = I so for example,
        upsample by R1 is the same with down sample by R2

     See also:  PPDEC"""

    # Parallelogram polyphase decomposition by simplifying sampling matrices
    # using the Smith decomposition of the quincunx matrices

    m, n = shape(p0)

    def type0():    # P1 = R1 * Q1 = D1 * R3
        x = zeros((2 * m, n))
        x[::2, :] = resamp(p0, 3)
        x[1::2, r_[1:len(x), 0]] = resamp(p1, 3)

        return x

    def type1():    # P2 = R2 * Q2 = D1 * R4
        x = zeros((2 * m, n))
        x[::2, :] = resamp(p0, 2)
        x[1::2, :] = resamp(p1, 2)

        return x

    def type2():    # P3 = R3 * Q2 = D2 * R1
        x = zeros((m, 2 * n))
        x[:, ::2] = resamp(p0, 1)
        x[r_[1:len(x), 0], 1::2] = resamp(p1, 1)

        return x

    def type3():    # P4 = R4 * Q1 = D2 * R2
        x = zeros((m, 2 * n))
        x[:, ::2] = resamp(p0, 0)
        x[:, 1::2] = resamp(p1, 0)

        return x

    def errhandler():
        print 'Invalid argument type'

    switch = {0: type0,
              1: type1,
              2: type2,
              3: type3}

    return switch.get(type, errhandler)()

# Support functions for generating filters


def ffilters(h0, h1):
    """ FFILTERS    Fan filters from diamond shape filters
    [f0, f1] = ffilters(h0, h1)"""

    f0 = [[None]] * 4
    f1 = [[None]] * 4

    # For the first half channels
    f0[0] = modulate2(h0, 'r')
    f1[0] = modulate2(h1, 'r')

    f0[1] = modulate2(h0, 'c')
    f1[1] = modulate2(h1, 'c')

    # For the second half channels,
    # use the transposed filters of the first half channels
    f0[2] = f0[0].T
    f1[2] = f1[0].T

    f0[3] = f0[1].T
    f1[3] = f1[1].T

    return f0, f1


def ld2quin(beta):
    """LD2QUIN    Quincunx filters from the ladder network structure
    Construct the quincunx filters from an allpass filter (beta) using the
    ladder network structure
    Ref: Phong et al., IEEE Trans. on SP, March 1995"""

    if beta.ndim > 1:
        print 'The input must be an 1-D filter'

    # Make sure beta is a row vector
    beta = beta.flatten(1)[:, newaxis].T

    lf = size(beta)
    n = lf / 2.0

    if n != floor(n):
        print 'The input allpass filter must be even length'

    # beta(z1) * beta(z2)
    sp = beta.T * beta

    # beta(z1*z2^{-1}) * beta(z1*z2)
    # Obtained by quincunx upsampling type 1 (with zero padded)
    h = qupz(sp, 1)

    # Lowpass quincunx filter
    h0 = h.copy()
    h0[2 * n, 2 * n] = h0[2 * n, 2 * n] + 1
    h0 = h0 / 2.0

    # Highpass quincunx filter
    h1 = -signal.convolve(h, h0)
    h1[4 * n - 1, 4 * n - 1] = h1[4 * n - 1, 4 * n - 1] + 1

    return h0, h1


def mctrans(b, t):
    """ MCTRANS McClellan transformation
    H = mctrans(B,T) produces the 2-D FIR filter H that
    corresponds to the 1-D FIR filter B using the transform T."""

    # Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    n = (size(b) - 1) / 2.0
    b = rot90(fftshift(rot90(b, 2)), 2)  # inverse fftshift

    a = c_[b[:, 0], 2 * b[:, 1:n + 1]]

    inset = floor((array(t.shape) - 1) / 2).astype(int)

    # Use Chebyshev polynomials to compute h
    P0, P1 = 1, t.copy()
    h = a[:, 1] * P1
    rows, cols = array([inset[0]]), array([inset[1]])
    h[rows, cols] = h[rows, cols] + a[:, 0] * P0
    for i in xrange(2, n + 1):
        P2 = 2 * signal.convolve(t, P1)
        rows = rows + inset[0]
        cols = cols + inset[1]
        P2[ix_(rows, cols)] = P2[ix_(rows, cols)] - P0
        rows = inset[0] + arange(0, P1.shape[0])
        cols = inset[1] + arange(0, P1.shape[1])
        hh = h.copy()
        h = a[:, i] * P2
        h[ix_(rows, cols)] = h[ix_(rows, cols)] + hh
        P0, P1 = P1.copy(), P2.copy()

    h = rot90(h, 2)  # Rotate for use with filter2
    return h


def modulate2(x, type, center=array([[0, 0]])):
    """ MODULATE2 2D modulation
    y = modulate2(x, type, [center])

    With TYPE = {'r', 'c' or 'b'} for modulate along the row, or column or
    both directions.
    CENTER especify the origin of modulation as
    floor(size(x)/2)+center(default is [0, 0])"""

    # Size and origin
    s = array([x.shape])
    o = floor(s / 2.0) + center

    n1 = array([arange(0, s[:, 0])]) - o[:, 0]
    n2 = array([arange(0, s[:, 1])]) - o[:, 1]

    def do_r():
        m1 = (-1)**n1
        y = x * tile(m1.T, (1, s[0, 1]))

        return y

    def do_c():
        m2 = (-1)**n2
        y = x * tile(m2, (s[0, 0], 1))

        return y

    def do_b():
        m1 = (-1)**n1
        m2 = (-1)**n2
        m = m1.T * m2
        y = x * m

        return y

    def errhandler():
        print 'Invalid input type'

    switch = {'r': do_r,
              'c': do_c,
              'b': do_b}

    return switch.get(str.lower(type[0]), errhandler)()


def reverse2(x):
    """ REVERSE2   Reverse order of elements in 2-d signal"""
    if x.ndim < 2:
        print 'Input must be a 2-D matrix.'
    return x[::-1, ::-1]

# Support fucntions to avoid visual distortion (used in DFB)


def backsamp(y):
    """ BACKSAMP
    Backsampling the subband images of the directional filter bank

       y = backsamp(y)

     Input and output are cell vector of dyadic length

     This function is called at the end of the DFBDEC to obtain subband images
     with overall sampling as diagonal matrices

     See also: DFBDEC"""

    # Number of decomposition tree levels
    n = int(log2(len(y)))

    if (n != round(n)) or (n < 1):
        print 'Input must be a cell vector of dyadic length'
    if n == 1:
        # One level, the decomposition filterbank shoud be Q1r
        # Undo the last resampling (Q1r = R1 * D0 * R2)
        for k in xrange(0, 2):
            y[k] = resamp(y[k], 3)
            y[k][:, 0::2] = resamp(y[k][:, 0::2], 0)
            y[k][:, 1::2] = resamp(y[k][:, 1::2], 0)
    elif n > 2:
        N = 2**(n - 1)
        for k in xrange(0, 2**(n - 2)):
            shift = 2 * (k + 1) - (2**(n - 2) + 1)
            # The first half channels
            y[2 * k] = resamp(y[2 * k], 2, shift)
            y[2 * k + 1] = resamp(y[2 * k + 1], 2, shift)
            # The second half channels
            y[2 * k + N] = resamp(y[2 * k + N], 0, shift)
            y[2 * k + 1 + N] = resamp(y[2 * k + 1 + N], 0, shift)

    return y


def rebacksamp(y):
    """ REBACKSAMP   Re-backsampling the subband images of the DFB

        y = rebacksamp(y)

     Input and output are cell vector of dyadic length

     This function is call at the begin of the DFBREC to undo the operation
     of BACKSAMP before process filter bank reconstruction.  In otherword,
     it is inverse operation of BACKSAMP

     See also:  BACKSAMP, DFBREC"""

    # Number of decomposition tree levels
    n = int(log2(len(y)))

    if (n != round(n)) or (n < 1):
        print 'Input must be a cell vector of dyadic length'
    if n == 1:
        # One level, the reconstruction filterbank shoud be Q1r
        # Redo the first resampling (Q1r = R1 * D0 * R2)
        for k in xrange(0, 2):
            y[k][:, ::2] = resamp(y[k][:, ::2], 1)
            y[k][:, 1::2] = resamp(y[k][:, 1::2], 1)
            y[k] = resamp(y[k], 2)
    elif n > 2:
        N = 2**(n - 1)
        for k in xrange(0, 2**(n - 2)):
            shift = 2 * (k + 1) - (2**(n - 2) + 1)
            # The first half channels
            y[2 * k] = resamp(y[2 * k], 2, -shift)
            y[2 * k + 1] = resamp(y[2 * k + 1], 2, -shift)
            # % The second half channels
            y[2 * k + N] = resamp(y[2 * k + N], 0, -shift)
            y[2 * k + 1 + N] = resamp(y[2 * k + 1 + N], 0, -shift)

    return y

# Other support functions


def smothborder(x, n):
    """
    SMTHBORDER  Smooth the borders of a signal or image
    y = smothborder(x, n)

    Input:
    x:      the input signal or image
    n:      number of samples near the border that will be smoothed

    Output:
    y:      output image

    Note: This function provides a simple way to avoid border effect."""

    # Hamming window of size 2N
    w = array([0.54 - 0.46 * cos(2 * pi * arange(0, 2 * n) / (2 * n - 1))])

    if x.ndim == 1:
        W = ones((1, x.size))
        W[:, 0:n] = w[:, 0:n]
        W[:, -1 - n + 1::] = w[:, -1 - n + 1::]
        y = W * x
    elif x.ndim == 2:
        n1, n2 = x.shape
        W1 = ones((n1, 1))
        W1[0:n] = w[:, 0:n].T
        W1[-1 - n + 1::] = w[:, n::].T

        y = tile(W1, (1, n2)) * x

        W2 = ones((1, n2))
        W2[:, 0:n] = w[:, 0:n]
        W2[:, -1 - n + 1::] = w[:, n::]
        y = tile(W2, (n1, 1)) * y
    else:
        print 'First input must be a signal or image'

    return y


def snr(im, est):
    """Encuentra la SNR entre la imagen de entrada (in) y la estimada (est) en
    decibeles.
    Referencia: Vetterly & Kovacevic, "Wavelets and Subband Coding", p. 386"""

    error = im - est
    r = 10 * log10(var(im) / mean(abs(error)**2))
    return r


#a = random.rand(1024,1024)
#a = arange(1,1025).reshape(32,32)
a = arange(1, 1025).reshape(32, 32)
#tic = time.clock()
#y = dup(a,array([2,2]),'m')
#toc = time.clock()
# print toc - tic
# print y
#b = qdown(y,2)
# print b
# pdb.set_trace()
y = dfbdec_l(a, 'pkva', 0)
print y
# pdb.set_trace()
z = dfbrec_l(y, 'pkva')
print z[31:], z.shape
print snr(a, z)
# 'haar': filterHaar,
#             'vk': filterVk,
#             'ko': filterKo,
#              'kos': filterKos,
#              'lax': filterLax,
#              'sk': filterSk,
#              'cd': filter79,
#              '7-9': filter79,
#              'pkva': filterPkva,
#              'pkva-half4': filterPkvaHalf4,
#              'pkva-half6': filterPkvaHalf6,
#              'pkva-half8': filterPkvaHalf8,
#              'oqf_362': filterOqf,
#              'test': filterTest,
#              'dvmlp': filterDvmlp,
#              'testDVM': filterDVM,
#              'qmf': filterQmf,
#              'qmf2': filterQmf2,
#              'sinc': filterSinc,
#              'dmaxflat4': filterDmaxflat4,
#              'dmaxflat5': filterDmaxflat5,
#              'dmaxflat6': filterDmaxflat6,
#              'dmaxflat7': filterDmaxflat7}

#h, g = pfilters('5-3')
# print h, h.shape
# print g, g.shape
# pdb.set_trace()
#k, l, m, n = wfb2dec(a, h, g)
# print k
# print l
# print m
# print n
# print wfb2rec(k,l,m,n,h,g)
#c, d = lpdec(a, h, g)
# print c
# print d
# print lprec(c, d, h, g)
