import os,logging

def setup(outdir = "logging"):
    os.makedirs(outdir,exist_ok=True)
    log_name = os.path.splitext(os.path.basename(__file__))[0]
    logging.basicConfig(
        level=logging.DEBUG,
        format='LINE %(lineno)-4d  %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename= os.path.join(outdir,log_name + ".log"),
        filemode='a+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('LINE %(lineno)-4d : %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return