import os
import pdb
from dotenv import load_dotenv
from .python_wer_evaluation import wer_calculation

# Load environment variables from .env file
load_dotenv('./evaluation/slr_eval/config.env')

# Retrieve environment variables
PREFIX = os.getenv('PREFIX', './')
MODE = os.getenv('MODE', 'dev')
EVALUATE_DIR = os.getenv('EVALUATE_DIR')
EVALUATE_PREFIX = os.getenv('EVALUATE_PREFIX')
OUTPUT_FILE = os.getenv('OUTPUT_FILE')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
PYTHON_EVALUATE = os.getenv('PYTHON_EVALUATE').lower() in ('true', '1', 't')
TRIPLET = os.getenv('TRIPLET').lower() in ('true', '1', 't')
SCLITE_PATH = os.getenv('SCLITE_PATH', './software/sclite')
PREPROCESS_FILES_PATH = os.getenv('PREPROCESS_FILES_PATH', './preprocess')

def evaluate(prefix=PREFIX, mode=MODE, evaluate_dir=EVALUATE_DIR, evaluate_prefix=EVALUATE_PREFIX,
             output_file=OUTPUT_FILE, output_dir=OUTPUT_DIR, python_evaluate=PYTHON_EVALUATE,
             triplet=TRIPLET, dataset = 'captar-libras_10_12'):
    '''
    TODO  change file save path
    '''
    print(os.getcwd())
    os.system(f"bash {evaluate_dir}/preprocess.sh {prefix + output_file} {prefix}tmp.ctm {prefix}tmp2.ctm")
    #os.system(f"cat {evaluate_dir}/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > {prefix}tmp.stm")
    #os.system(f"cat preprocess/{dataset}/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > {prefix}tmp.stm")
    os.system(f"cat {PREPROCESS_FILES_PATH.format(dataset = dataset)}/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > {prefix}tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python3 {evaluate_dir}/mergectmstm.py {prefix}tmp2.ctm {prefix}tmp.stm")
    os.system(f"cp {prefix}tmp2.ctm {prefix}out.{output_file}")
    if python_evaluate:
        #ret = wer_calculation(f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm", f"{prefix}out.{output_file}")
        ret = wer_calculation(f"preprocess/{dataset}/{evaluate_prefix}-{mode}.stm", f"{prefix}out.{output_file}")
        if triplet:
            wer_calculation(
                #f"{evaluate_dir}/{evaluate_prefix}-{mode}.stm",
                #f'preprocess/{dataset}/{evaluate_prefix}-{mode}.stm',
                f'{PREPROCESS_FILES_PATH.format(dataset = dataset)}/{evaluate_prefix}-{mode}.stm',
                f"{prefix}out.{output_file}",
                f"{prefix}out.{output_file}".replace(".ctm", "-conv.ctm")
            )
        return ret
    if output_dir is not None:
        if not os.path.isdir(prefix + output_dir):
            os.makedirs(prefix + output_dir)
        os.system(
            f"{SCLITE_PATH}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra -O {prefix + output_dir}"
        )
    else:
        os.system(
            f"{SCLITE_PATH}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra"
        )
    ret = os.popen(
        f"{SCLITE_PATH}  -h {prefix}out.{output_file} ctm "
        f"-r {prefix}tmp.stm stm -f 0 -o dtl stdout |grep Error"
    ).readlines()[0]
    return float(ret.split("=")[1].split("%")[0])

if __name__ == "__main__":
    evaluate()
    #evaluate("output-hypothesis-test.ctm", mode="test")
