import re

# CONSTANTS ####################################################################

ANSI_REGEX = r'\x1b\[[0-9;]*[mGKHF]'

# CLEAN ########################################################################

def clean(text: str, pattern: str=ANSI_REGEX, rewrite: str='') -> str:
    return re.sub(pattern=pattern, repl=rewrite, string=text)

# 1D => 2D #####################################################################

def split(text: str, height: int=-1, width: int=-1, separator: str='\n') -> list:
    # typically split on \n or at a fixed size
    __rows = text.split(separator) if separator else chunk(text, width)
    # :width would leave one character out when width == -1
    __width = slice(width if (width > 0) else None)
    # idem fro the height
    __height = slice(height if (height > 0) else None)
    # enforce the maximum dimensions
    return [__r[__width] for __r in __rows[__height] if __r]

# PAD ##########################################################################

def pad(rows: list, height: int=-1, width: int=-1, value: str='\x00') -> list:
    return [__r + (width - len(__r)) * value for __r in rows] + (height - len(rows)) * [width * value]

# RGB ENCODING #################################################################

def rgb_utf(rows: list) -> np.ndarray:
    __height, __width = len(rows), len(rows[0])
    # each character is encoded as 4 bytes
    __rows = [list(__r.encode('utf-32-be')) for __r in rows]
    # 2d reshaping
    __array = np.array(__rows, dtype=np.uint8).reshape((__height, __width, 4))
    # strip the leading byte, always null in utf-32 (big-endian)
    return __array[..., 1:]

# CUSTOM COLOR SCHEMES #########################################################

def mix_channels(channels: np.ndarray) -> np.ndarray:
    __mod = np.array(3 * [256], dtype=channels.dtype)
    __mix = [channels[0] + channels[-1], channels[1] + channels[-1], channels[-1]]
    return np.mod(__mix, __mod)

def rgb_mixed(rows: list) -> np.ndarray:
    return np.apply_along_axis(mix_channels, arr=rgb_utf(rows).astype(np.int32), axis=-1)

def rgb_hilbert(rows: list) -> np.ndarray:
    __height, __width = len(rows), len(rows[0])
    # each character is encoded as 4 bytes
    __rows = [[densecurves.hilbert.point(ord(__c), order=8, rank=3) for __c in __r] for __r in rows]
    # cast and reshape
    return np.array(__rows, dtype=np.uint8).reshape((__height, __width, 3))
