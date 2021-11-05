# -*- coding: utf-8 -*-
#    PyContourlet
#
#    A Python library for the Contourlet Transform.
#
#    Copyright (C) 2011-2020 Mazay Jiménez
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

import numpy as np
import math


def computescale(subband_dfb, ratio, start, end, mode):
    """
    COMPUTESCALE   Comupute display scale for PDFB coefficients

    computescale(subband_dfb, [ratio, start, end, mode])

    Input:

    subband_dfb:
        A multidimentional list, one for each layer of subband images from DFB.
    Each subband is represented as a numpy array
    ratio:
    Display ratio. It ranges from 1.2 to 10.

    start:
    Starting index of the cell vector subband_dfb for the computation.
    Its default value is 1.

    end:
    Ending index of the cell vector subband_dfb for the computation.
    Its default value is the length of subband_dfb.

    mode:
    coefficients mode (a string):
    'real' ----  Highpass filters use the real coefficients.
    'abs' ------ Highpass filters use the absolute coefficients.
    It's the default value
    Output:
    Scales ----- 1 X 2 list for two scales.

   See also:     SHOWPDFB"""

    if not isinstance(subband_dfb, list):
        print('Error in computescale.py! The first input must be a cell vector')

    # Display ratio
    if ratio is None:
        ratio = 2
    elif ratio < 1:
        print('Warning! the display ratio must be larger than 1!' +
              'Its defualt value is 2!')

    # Starting index for the cell vector subband_dfb
    if start is None:
        start = 0
    elif start < 0 or start > len(subband_dfb):
        print('Warning! The starting index from 0 to length(subband_dfb)!' +
              'Its defualt value is 0!')
        start = 0

    # Starting index for the cell vector subband_dfb
    if end is None:
        end = len(subband_dfb)
    elif end < 0 or end > len(subband_dfb):
        print('Warning! The ending index from 1 to length(subband_dfb)!' +
              'Its default value is length(subband_dfb)!')
        end = len(subband_dfb)

    # Coefficient mode
    if mode is None:
        mode = 'abs'
    elif mode != 'real' and mode != 'abs':
        print('Warning! There are only two coefficients mode: real, abs!' +
              'Its default value is "abs"!')
        mode = 'abs'

    # Initialization
    sum = 0
    mean = 0
    real_min = 1.0e14
    real_max = -1.0e14
    abs_min = 1.0e14
    abs_max = -1.0e14
    abs_sum = 0
    count = 0
    scales = np.zeros((1, 2))

    if mode == 'real':  # Use the real coefficients
        # Compute the mean of all coefficients
        for i in range(start, end):
            if isinstance(subband_dfb[i], list):
                m = len(subband_dfb[i])
                for j in range(m):

                    subband_min = subband_dfb[i][j].min()
                    if subband_min < real_min:
                        real_min = subband_min

                    subband_max = subband_dfb[i][j].max()
                    if subband_max > real_max:
                        real_max = subband_max

                    sum = sum + np.sum(subband_dfb[i][j])
                    count = count + subband_dfb[i][j].size
            else:
                subband_min = subband_dfb[i].min()
                if subband_min < real_min:
                    real_min = subband_min

                subband_max = subband_dfb[i].max()
                if subband_max > real_max:
                    real_max = subband_max

                sum = sum + np.sum(subband_dfb[i])
                count = count + subband_dfb[i].size

        if count < 2 or abs(sum) < 1e-10:
            print('Error in computescale.m! No data in this unit!')
        else:
            mean = sum / count

        # Compute the STD.
        sum = 0
        for i in range(start, end):
            if isinstance(subband_dfb[i], list):
                m = len(subband_dfb[i])
                for j in range(m):
                    sum = sum + sum((subband_dfb[i][j] - mean)**2)
            else:
                sum = sum + sum((subband_dfb[i] - mean)**2)

        std = math.sqrt(sum / (count - 1))

        scales[0] = max(mean - ratio * std, real_min)
        scales[1] = min(mean + ratio * std, real_max)

    else:  # Use the absolute coefficients
        # Compute the mean of absolute values
        for i in range(start, end):
            if isinstance(subband_dfb[i], list):
                m = len(subband_dfb[i])
                for j in range(m):

                    subband_min = abs(subband_dfb[i][j]).min()
                    if subband_min < abs_min:
                        abs_min = subband_min

                    subband_max = abs(subband_dfb[i][j]).max()
                    if subband_max > abs_max:
                        abs_max = subband_max

                abs_sum = abs_sum + np.sum(abs(subband_dfb[i][j]))
                count = count + subband_dfb[i][j].size
            else:
                subband_min = abs(subband_dfb[i]).min()
                if subband_min < abs_min:
                    abs_min = subband_min

                subband_max = abs(subband_dfb[i]).max()

                if subband_max > abs_max:
                    abs_max = subband_max

                abs_sum = abs_sum + np.sum(abs(subband_dfb[i]))
                count = count + subband_dfb[i].size

        if count < 2 or abs_sum < 1e-10:
            print('Error in computescale! No data in this unit!')
        else:
            abs_mean = abs_sum / count

        # Compute the std of absolute values
        sum = 0
        for i in range(start, end):
            if isinstance(subband_dfb[i], list):
                m = len(subband_dfb[i])
                for j in range(m):
                    sum = sum + np.sum((abs(subband_dfb[i][j]) - abs_mean)**2)
            else:
                sum = sum + np.sum((abs(subband_dfb[i]) - abs_mean)**2)

        std = math.sqrt(sum / (count - 1))

        # Compute the scale values
        scales[0] = max(abs_mean - ratio * std, abs_min)
        scales[1] = min(abs_mean + ratio * std, abs_max)

    return scales
