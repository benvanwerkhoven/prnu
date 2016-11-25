import os


def read_kernel(filename):
    """ Return the kernel specified by filename as string """
    path = "/".join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
    file = path+'/prnu/kernels/'+filename
    with open(file, 'r') as f:
        return f.read()

#for debuggin Python code
def interactive():
    import readline
    import rlcompleter
    readline.parse_and_bind("tab: complete")
    import code
    code.interact(local=dict(globals(), **locals()) )


