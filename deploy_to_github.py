# -*- coding: utf-8 -*-
"""
GitHub一键部署脚本
- 创建新仓库
- 上传监控系统所有文件
- 配置飞书Webhook Secret
"""
import requests
import base64
import os
import json

GITHUB_API = "https://api.github.com"

# 需要上传的文件列表（监控系统相关）
FILES_TO_UPLOAD = [
    "realtime_monitor.py",
    "feishu_notifier.py",
    "run_monitor.bat",
    "test_feishu.py",
    "requirements.txt",
    "README_monitor.md",
    ".github/workflows/monitor.yml",
    "config.py",
    "data_fetcher.py",
    "pattern_recognizer.py",
    "strategy.py",
    "market_regime.py",
    "train_ml_from_trades.py",
]

# 需要创建的目录
DIRS = [".github/workflows", "data_cache"]

def get_token():
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        token = input("请输入GitHub Personal Access Token: ").strip()
    return token

def get_headers(token):
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }

def create_repo(token, repo_name, private=True):
    """创建新仓库"""
    url = f"{GITHUB_API}/user/repos"
    data = {
        "name": repo_name,
        "description": "A股形态突破策略实时监控系统",
        "private": private,
        "auto_init": False
    }
    resp = requests.post(url, headers=get_headers(token), json=data)
    if resp.status_code == 201:
        print(f"✅ 仓库创建成功: {resp.json()['html_url']}")
        return resp.json()
    elif resp.status_code == 422:
        print(f"⚠️ 仓库已存在，直接使用")
        # 获取仓库信息
        url2 = f"{GITHUB_API}/repos/{get_username(token)}/{repo_name}"
        resp2 = requests.get(url2, headers=get_headers(token))
        if resp2.status_code == 200:
            return resp2.json()
    print(f"❌ 创建失败: {resp.status_code} {resp.text}")
    return None

def get_username(token):
    url = f"{GITHUB_API}/user"
    resp = requests.get(url, headers=get_headers(token))
    return resp.json()['login']

def upload_file(token, owner, repo, filepath, branch="main"):
    """上传单个文件（带重试）"""
    import time
    full_path = os.path.join(os.path.dirname(__file__), filepath)
    if not os.path.exists(full_path):
        print(f"  ⚠️ 文件不存在，跳过: {filepath}")
        return False

    with open(full_path, 'rb') as f:
        content = base64.b64encode(f.read()).decode()

    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{filepath}"

    for retry in range(3):
        try:
            # 检查文件是否已存在
            resp = requests.get(url, headers=get_headers(token), params={"ref": branch}, timeout=30)
            sha = resp.json().get('sha') if resp.status_code == 200 else None

            data = {
                "message": f"Add {filepath}",
                "content": content,
                "branch": branch
            }
            if sha:
                data['sha'] = sha
                data['message'] = f"Update {filepath}"

            resp = requests.put(url, headers=get_headers(token), json=data, timeout=30)
            if resp.status_code in [200, 201]:
                print(f"  ✅ {filepath}")
                return True
            else:
                print(f"  ❌ {filepath}: {resp.status_code}")
                return False
        except Exception as e:
            if retry < 2:
                time.sleep(3)
            else:
                print(f"  ❌ {filepath}: 连接失败 {e}")
                return False
    return False

def set_secret(token, owner, repo, secret_name, secret_value):
    """设置GitHub Actions Secret"""
    from nacl import encoding, public

    # 获取仓库公钥
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/secrets/public-key"
    resp = requests.get(url, headers=get_headers(token))
    if resp.status_code != 200:
        print(f"❌ 获取公钥失败: {resp.status_code}")
        return False
    pub_key = resp.json()
    key_id = pub_key['key_id']
    key = public.PublicKey(pub_key['key'].encode(), encoding.Base64Encoder())

    # 加密
    sealed_box = public.SealedBox(key)
    encrypted = sealed_box.encrypt(secret_value.encode())
    encrypted_value = base64.b64encode(encrypted).decode()

    # 设置Secret
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/secrets/{secret_name}"
    data = {
        "encrypted_value": encrypted_value,
        "key_id": key_id
    }
    resp = requests.put(url, headers=get_headers(token), json=data)
    if resp.status_code in [200, 201]:
        print(f"✅ Secret设置成功: {secret_name}")
        return True
    else:
        print(f"❌ Secret设置失败: {resp.status_code} {resp.text}")
        return False

def main():
    print("=" * 60)
    print("GitHub一键部署 - A股策略实时监控")
    print("=" * 60)

    token = get_token()
    if not token:
        print("❌ 需要Token")
        return

    # 验证Token
    username = get_username(token)
    print(f"\n当前用户: {username}")

    repo_name = input("仓库名称 (默认: a-stock-monitor): ").strip() or "a-stock-monitor"
    webhook = input("飞书Webhook地址 (直接回车跳过): ").strip()

    # 创建仓库
    print(f"\n创建仓库: {repo_name}")
    repo = create_repo(token, repo_name, private=True)
    if not repo:
        return
    owner = repo['owner']['login']

    # 上传文件
    print(f"\n上传文件...")
    success = 0
    for filepath in FILES_TO_UPLOAD:
        if upload_file(token, owner, repo_name, filepath):
            success += 1
    print(f"\n上传完成: {success}/{len(FILES_TO_UPLOAD)} 个文件")

    # 设置Secret
    if webhook:
        print(f"\n配置飞书Webhook Secret...")
        try:
            set_secret(token, owner, repo_name, "FEISHU_WEBHOOK", webhook)
        except ImportError:
            print("⚠️ 需要安装pynacl: pip install pynacl")
            print("  或手动在仓库 Settings → Secrets → Actions 中添加 FEISHU_WEBHOOK")

    print(f"\n{'='*60}")
    print(f"✅ 部署完成！")
    print(f"仓库地址: {repo['html_url']}")
    print(f"Actions地址: {repo['html_url']}/actions")
    print(f"\n下一步:")
    print(f"1. 进入仓库 Settings → Secrets → Actions")
    print(f"2. 确认 FEISHU_WEBHOOK 已设置")
    print(f"3. 进入 Actions 页面，启用 workflow")
    print(f"4. 点击 'Run workflow' 手动测试一次")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
