import logging

logging.basicConfig(level=logging.INFO)

def logprint(*args, **kwargs):
    logging.info(" ".join(str(a) for a in args))