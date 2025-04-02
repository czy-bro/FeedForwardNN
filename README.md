
# README

## 项目简介

本项目使用 PyTorch 实现了一个前馈神经网络（FeedForward Neural Network, FFNN）来拟合目标函数：
\(y = \log_2(x) + \cos(\pi x / 2)\)

## 代码结构

- **数据生成 (****`generate_data`****)**: 生成 2000 个样本数据，并划分为训练集、验证集和测试集。
- **数据可视化**: 通过 `matplotlib` 绘制数据集分布。
- **神经网络 (****`FeedForwardNN`****)**: 搭建包含多个隐藏层的前馈神经网络。
- **训练 (****`train`****)**: 使用 MSE 损失函数和 Adam 优化器进行模型训练。
- **验证 (****`validate`****)**: 在验证集上计算均方误差 (MSE) 并输出预测结果。
- **测试 (****`evaluate`****)**: 在测试集上评估最终模型性能。

## 依赖项

请确保安装了以下 Python 库：

```bash
pip install torch numpy matplotlib scikit-learn
```

## 运**行方**式

1. **运行数据生成和可视化**

   ```python
   python main.py
   ```

   生成数据并绘制数据分布图。

2. **训练模型**

   ```python
   model = FeedForwardNN(input_dim=1, hidden_dim=64, output_dim=1, num_layers=3, activation=nn.ReLU)
   train(model, x_train, y_train, epochs=100, lr=0.001)
   ```

   训练神经网络，并每 100 轮打印一次损失。

3. **验证与测试模型**

   ```python
   mse_val, predictions_val = validate(model, x_val, y_val)
   mse_test, predictions_test = evaluate(model, x_test, y_test)
   ```

   计算模型在验证集和测试集上的 MSE，并输出测试结果。

## 结果示例

```
Epoch[100/100], Loss: 0.0156
Validation MSE: 0.0123
Test MSE: 0.0137
```

## 可能的问题与改进

- 训练过程中可能会遇到梯度消失或梯度爆炸的问题，可以调整 `hidden_dim` 或 `num_layers`。
- 选择不同的 `activation`（如 `nn.Tanh` 或 `nn.Sigmoid`）可能会影响模型表现。
- 通过 `learning rate decay` 或 `dropout` 可以提升模型泛化能力。

## 许可证

本项目基于 MIT 许可证开源，可自由使用和修改。

