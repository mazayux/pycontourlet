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

from resamp import resamp
import math

def backsamp(y):
    """ BACKSAMP
    Backsampling the subband images of the directional filter bank

       y = backsamp(y)

     Input and output are lists of dyadic length

     This function is called at the end of the DFBDEC to obtain subband images
     with overall sampling as diagonal matrices

     See also: DFBDEC"""

    # Number of decomposition tree levels
    n = int(math.log2(len(y)))

    if n < 1:
        print('Input must be a cell vector of dyadic length')
    elif n == 1:
        # One level, the decomposition filterbank shoud be Q1r
        # Undo the last resampling (Q1r = R2 * D1 * R3)
        for k in range(2):
            y[k] = resamp(y[k], 3, None, None)
            y[k][:, ::2] = resamp(y[k][:, ::2], 0, None, None)
            y[k][:, 1::2] = resamp(y[k][:, 1::2], 0, None, None)
    else:
        N = 2**(n - 1)
        for k in range(2**(n - 2)):
            shift = 2 * (k + 1) - (2**(n - 2) + 1)

            # The first half channels
            y[2*k] = resamp(y[2*k], 2, shift, None)
            y[2*k+1] = resamp(y[2*k+1], 2, shift, None)

            # The second half channels
            y[2*k+N] = resamp(y[2*k+N], 0, shift, None)
            y[2*k+1+N] = resamp(y[2*k+1+N], 0, shift, None)

    return y
