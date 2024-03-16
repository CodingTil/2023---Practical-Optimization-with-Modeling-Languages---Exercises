#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 22:56:53 2020

@author: luebbecke
"""

# hard

n = 200  # items
m = 150  # bins

b = 100  # capacity
a = [
    26,
    10,
    76,
    75,
    42,
    9,
    13,
    92,
    73,
    61,
    92,
    80,
    61,
    76,
    28,
    83,
    95,
    38,
    76,
    72,
    19,
    37,
    20,
    18,
    22,
    26,
    89,
    63,
    29,
    38,
    42,
    5,
    5,
    28,
    52,
    52,
    27,
    18,
    44,
    8,
    42,
    71,
    28,
    72,
    81,
    39,
    83,
    73,
    82,
    36,
    54,
    46,
    50,
    66,
    59,
    65,
    21,
    52,
    74,
    86,
    11,
    18,
    72,
    12,
    53,
    81,
    9,
    23,
    7,
    95,
    90,
    54,
    47,
    45,
    13,
    4,
    67,
    70,
    91,
    88,
    58,
    25,
    11,
    3,
    44,
    45,
    87,
    30,
    17,
    46,
    18,
    75,
    54,
    18,
    74,
    82,
    41,
    22,
    27,
    28,
    75,
    40,
    7,
    8,
    57,
    4,
    6,
    46,
    13,
    11,
    30,
    36,
    87,
    42,
    80,
    39,
    6,
    40,
    67,
    12,
    16,
    41,
    4,
    22,
    77,
    70,
    47,
    61,
    90,
    17,
    76,
    25,
    87,
    73,
    87,
    44,
    17,
    60,
    36,
    16,
    2,
    82,
    95,
    75,
    1,
    94,
    81,
    40,
    92,
    15,
    62,
    88,
    85,
    38,
    92,
    56,
    81,
    94,
    25,
    76,
    5,
    8,
    1,
    63,
    66,
    57,
    3,
    72,
    79,
    12,
    26,
    63,
    3,
    38,
    93,
    76,
    7,
    60,
    38,
    55,
    9,
    9,
    32,
    30,
    50,
    21,
    75,
    29,
    90,
    43,
    65,
    58,
    50,
    56,
    58,
    10,
    44,
    6,
    14,
    80,
]


import binpackingmodel

# binpackingmodel.solve(m,a,b)


import makespanscheduling

makespanscheduling.solve(m, a, b)
