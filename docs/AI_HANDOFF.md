# handle-swap-app：项目运行与AI接手

Flask拉手替换工具：区域选取、透明抠图、保留比例的本地合成，以及可选云端图像编辑。

## 仓库身份

- 分类：电商与图像；类型：python。
- 默认分支：`main`；当前可见性：公开。
- 本说明针对本仓库当前文件；历史本机绝对路径与相似项目不能作为当前运行位置。

## 先读这些文件

- [app.py](<../app.py>)
- [index.html](<../index.html>)
- [test_image_ops.py](<../test_image_ops.py>)
- [test_composite.py](<../test_composite.py>)

## 安装和启动

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -p "test_*.py"
python app.py
```

本机地址：http://127.0.0.1:5000。端口以启动输出为准；手机访问须使用运行电脑当前局域网IPv4与允许的专用网络端口。



## 配置名称

从源码及示例提取到的环境变量名称：

`GEMINI_API_KEY`、`OPENAI_API_KEY`、`PORT`。

这些只是名称，不是已配置的值；有些是工具或部署变量，并非全部必填。优先读.env.example及引用处，不把真实密钥写进源码或提交。

## 数据与搬迁

上传图与密钥不属于源码；用真实新拉手照片保留外形、比例和安装位置。可选云接口另装requirements-ai.txt。

## 怎么验证

先执行 [验证记录](VERIFICATION.md) 所列命令，再按改动检查实际页面或输出。没有测试脚本时不能把占位npm test当作通过。仅文档更改检查链接和路径；涉及数据、几何、金额或接口的修改运行相应回归测试。

- 本轮以合成图片验证本地HTTP接口和几何约束，没有提交付费云端请求。

## 继续维护

先读根目录AGENTS.md；若它是CLAUDE.md的链接则直接读后者。保留已有项目约定。修改前确认分支、当前差异、数据目录与实际服务。完成后更新本说明/验证记录，提交明确的源码和文档路径，并核对GitHub远端提交。公开状态、旧版、URL与业务数据需按用户既有约定保持。
