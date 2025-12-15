# 雪球模拟盘接入指南

## 准备工作

### 1. 注册雪球账号并创建组合

1. 访问 [雪球网](https://xueqiu.com/)
2. 注册账号（使用手机号或邮箱）
3. 登录后，点击"交易" -> "模拟炒股"
4. 创建一个新的模拟组合
5. 记录组合代码（在组合页面URL中，格式如 `ZH123456`）

### 2. 安装依赖

已完成！easytrader 和 schedule 已安装成功。

```bash
pip install easytrader schedule
```

## 测试连接

### 重要说明

⚠️ **雪球登录方式已变更**：easytrader 连接雪球需要使用 **Cookies**，不能直接使用用户名和密码登录。

### 步骤一：获取雪球 Cookies

#### 方法一：使用浏览器开发者工具（推荐）

1. 打开 **Chrome** 或 **Edge** 浏览器
2. 访问 https://xueqiu.com/ 并登录你的账号
3. 登录成功后，按 **F12** 键打开开发者工具
4. 点击顶部的 **Network**（网络）标签
5. 按 **F5** 刷新页面
6. 在左侧请求列表中选择任意一个请求（如 "xueqiu.com"）
7. 在右侧查看 **Request Headers**（请求头）
8. 找到 **Cookie:** 这一行
9. 复制 Cookie: 后面的**所有内容**（很长一串）

Cookie 示例：
```
xq_a_token=abc123def456...; xq_r_token=xyz789...; device_id=...; ...
```

#### 方法二：使用浏览器插件

1. 安装 Chrome 插件 **EditThisCookie** 或 **Cookie-Editor**
2. 在雪球网站登录后，点击插件图标
3. 选择 "Export" 导出所有 cookies
4. 保存导出的内容

### 步骤二：运行测试脚本

运行测试脚本验证能否连接到雪球：

```bash
python scripts/live/test_xueqiu_connection.py
```

按照提示：
1. 确认是否已获取 Cookies（输入 y）
2. 粘贴从浏览器复制的完整 Cookie 字符串
3. 输入组合代码（如 ZH123456）

如果连接成功，脚本会：
1. 显示当前持仓信息
2. 询问是否保存配置到 `config/xueqiu_account.txt`（建议保存，方便后续使用）

> ⚠️ **重要提示**：
> - Cookie 有效期通常为几天到几周，过期后需要重新获取
> - 账号配置文件已加入 `.gitignore`，不会被上传到GitHub
> - 切勿分享你的 Cookie 给他人


## 常见问题

### Q1: 提示"Cookie已过期"或连接失败

**可能原因**：
- Cookie 过期（通常几天到几周有效）
- Cookie 复制不完整
- 网络问题

**解决方法**：
1. 重新登录雪球网页版
2. 按照上述步骤重新获取最新的 Cookie
3. 确保复制了完整的 Cookie 字符串（包含 xq_a_token、xq_r_token 等）
4. 检查网络连接

### Q2: 提示"组合代码错误"

**解决方法**：
1. 登录雪球网页版
2. 进入你的模拟组合页面
3. 从URL中复制组合代码（如 `https://xueqiu.com/P/ZH123456` 中的 `ZH123456`）
4. 注意大小写敏感

### Q3: 如何找到组合代码？

1. 登录雪球
2. 点击 "交易" -> "我的组合"
3. 选择你要使用的组合
4. 查看浏览器地址栏，URL格式为 `https://xueqiu.com/P/ZH******`
5. `ZH******` 就是你的组合代码

### Q4: Cookie 会过期吗？

是的，Cookie 有有效期（通常几天到几周）。过期后需要：
1. 重新登录雪球网页版
2. 重新获取 Cookie
3. 更新配置文件或重新运行测试脚本

### Q5: easytrader版本问题

如果遇到接口报错，可能是雪球API变更，尝试更新easytrader：

```bash
pip install --upgrade easytrader
```


## 下一步

连接成功后，我们将开发：
1. 自动读取推荐列表（从 `daily_picks`）
2. 自动生成交易计划（T+1买入，T+5卖出）
3. 自动执行交易（通过雪球API下单）

## 参考资料

- [easytrader 官方文档](https://github.com/shidenggui/easytrader)
- [雪球模拟炒股](https://xueqiu.com/snowman)
