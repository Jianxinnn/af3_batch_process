
import subprocess
from loguru import logger

def run_command(command, placeholder=None):
    """
    通用命令执行函数
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
    output_lines = []
    for line in iter(process.stdout.readline, ''):
        if line:
            output_lines.append(line)
            logger.debug(line.strip())
            if placeholder is not None:
                placeholder.markdown(f"```\n{''.join(output_lines)}\n```")
    process.stdout.close()
    process.wait()
    return ''.join(output_lines)

# 保持原有的run_alphafold函数作为特例
def run_alphafold(command, placeholder=None):
    return run_command(command, placeholder)
