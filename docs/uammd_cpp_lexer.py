
from pygments.lexer import *
from pygments.token import *
from  pygments.lexers.c_cpp import *

__all__ = ['UAMMDCLexer', 'UAMMDCppLexer']

class UAMMDCppLexer(CppLexer):
    name = 'UAMMDC++'
    aliases = ['ucpp', 'uc++']

    tokens = {
        'statements': [
            (r'(Box|real4|real|real3|real2|int2|int3|int4)\b', Keyword.Type),
            (words(('__global__','__device__','__host__'), suffix=r'\b'), Keyword.Reserved),
            inherit,
        ]
    }
