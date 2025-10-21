import subprocess
import os
import sys
from huggingface_hub import snapshot_download

def setup_autodl_proxy():
    """
    设置 AutoDL 的网络代理环境变量，用于加速。
    """
    print("正在设置 AutoDL 网络加速...")
    try:
        # 运行 bash 命令并捕获输出
        # 注意: AutoDL 的加速脚本可能是 /etc/network_proxy 或 /etc/network_turbo
        result = subprocess.run(

            'bash -c "source /etc/network_turbo && env | grep proxy"', 

            shell=True, 

            capture_output=True, 

            text=True
        )
        
        output = result.stdout
        # 解析输出并设置环境变量
        for line in output.splitlines():
            if '=' in line:
                var, value = line.split('=', 1)
                os.environ[var.strip()] = value.strip()
                print(f"已设置环境变量: {var.strip()}")
        print("网络加速设置成功。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"网络加速脚本执行失败: {e.stderr}")
        return False
    except Exception as e:
        print(f"设置网络加速时出错: {e}")
        return False

def download_model():
    """
    下载指定的 Hugging Face 模型。
    """
    # 1. 设置模型和下载参数
    # --- 请在这里修改为您想下载的模型 ---
    model_repo_id = "circulus/alpaca-7b" # 示例：Llama-2 7B 模型
    
    # 定义基础存储路径
    base_storage_path = "/root/autodl-tmp"
    # 根据模型ID自动生成清晰的本地文件夹名
    local_dir = os.path.join(base_storage_path, model_repo_id.replace("/", "_"))

    # 从环境变量读取 Hugging Face Token (更安全)
    hf_token = os.getenv("HUGGING_FACE_TOKEN")

    if not hf_token:
        print("错误: 未找到 Hugging Face Token。请设置 HUGGING_FACE_HUB_TOKEN 环境变量。", file=sys.stderr)
        return

    print("-" * 60)
    print(f"准备下载模型: {model_repo_id}")
    print(f"将要保存到: {local_dir}")
    print("-" * 60)

    # 确保目标文件夹存在
    os.makedirs(local_dir, exist_ok=True)

    try:
        # 2. 执行下载
        snapshot_download(
            repo_id=model_repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # 在AutoDL上建议设为False
            token=hf_token,
            resume_download=True # 支持断点续传
        )
        print("\n模型下载成功！")
        print(f"模型文件已保存至: {local_dir}")
        
    except Exception as e:
        print(f"\n下载模型时发生错误: {e}", file=sys.stderr)
        print("请检查：")
        print(f"1. 您的Hugging Face Token是否有权限访问 '{model_repo_id}'。")
        print("2. 网络连接是否正常 (代理是否生效)。")
        print("3. 磁盘空间是否充足。")

if __name__ == "__main__":
    if setup_autodl_proxy():
        download_model()