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
from scipy import weave
from scipy.weave import converters
from matplotlib.cbook import is_string_like
#import time


def resampc(x, type, shift, extmod):
    """ RESAMPC	Resampling along the column

    y = resampc(x, type, shift, extmod)

    Input:
    x:	image that is extendable along the column direction
    type:	either 0 or 1 (0 for shuffering down and 1 for up)
    shift:	amount of shifts (typically 1)
    extmod: extension mode:
    'per' 	periodic
    'ref1'	reflect about the edge pixels
    'ref2'	reflect, doubling the edge pixels

    Output:
    y:	resampled image with:
    R1 = [1, shift; 0, 1] or R2 = [1, -shift; 0, 1]"""

    if type != 0 and type != 1:
        print "The second input (type) must be either 1 or 2"
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
           for (j = 0; j < n; j++){
	    /* Circular shift in each column */
	    if (type == 0)
		k = (s * j) % m;
	    else
		k = (-s * j) % m;

	    /* Convert to non-negative mod if needed */
	    if (k < 0)
		k += m;

	    for (i = 0; i < m; i++){
		if (k >= m)
		    k -= m;

		y(i, j) = x(k, j);

		k++;
	    }
	}
      """
    # call weave
    weave.inline(code, ['m', 'n', 'x', 'y', 's', 'type'],
                 type_converters=converters.blitz, compiler='gcc')
    return y

#x  =  arange(1,26).reshape(5,5).T
#x = random.rand(4096,4096)
#type = 2
#shift  = 1
#extmod = 'per'
#tic = time.clock()
#y = resampc(x, type, shift, extmod)
#toc = time.clock()
# print toc - tic
# print y
