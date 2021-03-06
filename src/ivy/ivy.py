#! /usr/bin/env python
#
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
import sys
import pickle
import string
from ivy_compiler import IvyError, ivy_new, ivy_load_file
from ivy_ui import ui_main_loop
from ivy_utils import Parameter, set_parameters
import ivy_logic
import proof as pf
import ivy_utils as iu
import ivy_module
#import tactics_api as ta

# mode = Parameter("mode",None)

def usage():
    print "usage: \n  {} <file>.[a2g,ivy,dfy]\n {} <file>.a2g <file.[ivy,dfy]> ".format(sys.argv[0],sys.argv[0])
    sys.exit(1)

def open_read(fn):
    f = open(fn,'r')
    if not f:
        print "not found: %s" % fn
        sys.exit(1)
    return f

def read_params():
    ps = dict()
    args = sys.argv[1:]
    while args and '=' in args[0]:
        thing = string.split(args[0],'=')
        if len(thing) > 2:
            usage()
        ps[thing[0]] = thing[1]
        args = args[1:]
    try:
        set_parameters(ps)
    except IvyError as e:
        print e
        exit(1)
    sys.argv = sys.argv[0:1] + args

# def ivy_init2():
#     if len(sys.argv) < 2 or len(sys.argv) > 2:
#         usage()
#     fn = sys.argv[1]
#     try:
#         session = pf.AnalysisSession(fn)
#         ta.set_context(session)
# # tt.CheckCover.apply.im_func.func_code.co_argcount
#         return session.analysis_state.ivy_ag
#     except IvyError as e:
#         e.filename = fn
#         print repr(e)
#         sys.exit(1)
    
    

def ivy_init():
    read_params()

#    if mode.get() == "ivy2":
#        return ivy_init2()

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        usage()

    files = [(fn,open_read(fn)) for fn in sys.argv[1:]]

    if files[0][0].endswith('.a2g'):
        fn,f = files.pop(0)
        ag = pickle.load(f)

        if not hasattr(ag.domain,'all_relations'):
#            print "reconstructing all_relations"
            ag.domain.all_relations = []
            for x in ag.domain.relations:
                ag.domain.all_relations.append((x,ag.domain.relations[x]))
            for (x,y) in ag.domain.concepts:
                ag.domain.all_relations.append((x.atom.relname,len(x.atom.args)))
        ivy_logic.sig = ag.domain.sig # TODO: make this an environment
        f.close()
    else:
#        print "creating new"
        ag = ivy_new()

    if files:
        fn,f = files.pop(0)
        if not fn.endswith('.ivy') and not fn.endswith('.dfy'):
            usage()
        try:
#            print "loading file %s" % fn
            with iu.SourceFile(fn):
                ivy_load_file(f,ag)
        except IvyError as e:
            if not hasattr(e,'filename'):
                e.filename = fn
            print repr(e)
            sys.exit(1)
    return ag

if __name__ == "__main__":
    with ivy_module.Module():
        ui_main_loop(ivy_init())


