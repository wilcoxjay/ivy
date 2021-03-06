#
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
import ply.lex as lex

tokens = (
   'COMMA',
   'LPAREN',
   'RPAREN',
   'PLUS',
   'TIMES',
   'TILDA',
   'AND',
   'OR',
   'EQ',
   'TILDAEQ',
   'SEMI',
   'ASSIGN',
   'DOT',
   'LCB',
   'RCB',
   'ARROW',
   'IFF',
   'SYMBOL',
   'VARIABLE',
   'COLON',
   'LE',
   'LT',
   'GE',
   'GT',
   'MINUS',
)

reserved = all_reserved = {
   'relation' : 'RELATION',
   'individual' : 'INDIV',
   'axiom' : 'AXIOM',
   'conjecture' : 'CONJECTURE',
   'schema' : 'SCHEMA',
   'instantiate' : 'INSTANTIATE',
   'derived' : 'DERIVED',
   'concept' : 'CONCEPT',
   'init' : 'INIT',
   'action' : 'ACTION',
   'state' : 'STATE',
   'assume' : 'ASSUME',
   'assert' : 'ASSERT',
   'set' : 'SET',
   'null' : 'NULL',
   'old' : 'OLD',
   'from' : 'FROM',
   'update' : 'UPDATE',
   'params' : 'PARAMS',
   'in' : 'IN',
   'match' : 'MATCH',
   'ensures' : 'ENSURES',
   'requires' : 'REQUIRES',
   'modifies' : 'MODIFIES',
   'true' : 'TRUE',
   'false' : 'FALSE',
   'fresh' : 'FRESH',
   'module' : 'MODULE',
   'type' : 'TYPE',
   'if' : 'IF',
   'else' : 'ELSE',
   'local' : 'LOCAL',
   'let' : 'LET',
   'call' : 'CALL',
   'entry' : 'ENTRY',
   'macro' : 'MACRO',
   'interpret' : 'INTERPRET',
   'forall' : 'FORALL',
   'exists' : 'EXISTS',
   'returns' : 'RETURNS',
   'mixin' : 'MIXIN',
   'before' : 'BEFORE',
   'after' : 'AFTER',
   'isolate' : 'ISOLATE',
   'with' : 'WITH',
   'export' : 'EXPORT',
   'delegate' : 'DELEGATE',
   'import' : 'IMPORT',
   'include' : 'INCLUDE',
}

tokens += tuple(all_reserved.values())


t_TILDA    = r'\~'
t_COMMA    = r'\,'
t_PLUS    = r'\+'
t_TIMES   = r'\*'
t_MINUS   = r'\-'
t_LT      = r'\<'
t_LE      = r'\<='
t_GT      = r'\>'
t_GE      = r'\>='
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_OR = r'\|'
t_AND = r'\&'
t_EQ = r'\='
t_TILDAEQ = r'\~='
t_SEMI = r'\;'
t_ASSIGN = r'\:='
t_DOT = r'\.'
t_LCB  = r'\{'
t_RCB  = r'\}'
t_ARROW = r'\->'
t_IFF = r'\<->'
t_COLON = r':'

t_ignore  = ' \t'
t_ignore_COMMENT = r'\#.*'

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

def t_SYMBOL(t):
    r'[_a-z0-9][_a-zA-Z0-9]*(\[[ab-zA-Z_0-9]*\])*'
    t.type = reserved.get(t.value,'SYMBOL')
    return t

def t_VARIABLE(t):
    r'[A-Z][_a-zA-Z0-9]*(\[[ab-zA-Z_0-9]*\])*'
    t.type = reserved.get(t.value,'VARIABLE')
    return t

def t_error(t):
    print "Illegal character '%s'" % t.value[0]
    t.lexer.skip(1)

lexer = lex.lex()

class LexerVersion(object):
    """ Context Manager that sets the lexer based on the given language version
    """
    def __init__(self,version):
        self.version = version
    def __enter__(self):
        global reserved
        self.orig_reserved = reserved
        reserved = dict(all_reserved)
#        print "version {}".format(self.version)
        if self.version <= [1,0]:
            for s in ['state','local']:
#                print "deleting {}".format(s)
                del reserved[s]
        if self.version <= [1,1]:
            for s in ['returns','mixin','before','after','isolate','with','export','delegate','import','include']:
#                print "deleting {}".format(s)
                if s in reserved:
                    del reserved[s]
        else:
            for s in ['state','set','null','match']:
                if s in reserved:
                    del reserved[s]
        return self
    def __exit__(self,exc_type, exc_val, exc_tb):
        global reserved
        reserved = self.orig_reserved
        return False # don't block any exceptions
