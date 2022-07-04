/**
 * 线性回归的一个demo
 * 回归表示模型会输出实数值
 *
 * 预测文件下载时间
 *
 */

const tf = require("@tensorflow/tfjs-node")

const trainData = {
    sizeMB:  [0.080, 9.000, 0.001, 0.100, 8.000, 5.000, 0.100, 6.000, 0.050, 0.500,
        0.002, 2.000, 0.005, 10.00, 0.010, 7.000, 6.000, 5.000, 1.000, 1.000],
    timeSec: [0.135, 0.739, 0.067, 0.126, 0.646, 0.435, 0.069, 0.497, 0.068, 0.116,
        0.070, 0.289, 0.076, 0.744, 0.083, 0.560, 0.480, 0.399, 0.153, 0.149]
};
const testData = {
    sizeMB:  [5.000, 0.200, 0.001, 9.000, 0.002, 0.020, 0.008, 4.000, 0.001, 1.000,
        0.005, 0.080, 0.800, 0.200, 0.050, 7.000, 0.005, 0.002, 8.000, 0.008],
    timeSec: [0.425, 0.098, 0.052, 0.686, 0.066, 0.078, 0.070, 0.375, 0.058, 0.136,
        0.052, 0.063, 0.183, 0.087, 0.066, 0.558, 0.066, 0.068, 0.610, 0.057]
};

// 训练张量的特征值
let trainXs = tf.tensor(trainData.sizeMB)
// 训练张量的目标值
let trainYs = tf.tensor(trainData.timeSec)

// 创建一个模型
const model = tf.sequential();
// 添加密集层， 输入形状是 一维矩阵， 输出结果的长度为1的向量
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// 输出模型的结构
model.summary();

// 编译模型
model.compile({
    optimizer: tf.train.sgd(0.0002), // 使用随机梯度下降 优化器
    loss: tf.losses.meanSquaredError // 平均绝对值误差作为损失函数
});

(async function (){
    await model.fit(trainXs, trainYs, {
        epochs: 1000, // 训练1000次，权重会更新 1000次
        callbacks: {

        }
    })
    /**
     * 评估学习成果
     * 一定要有测试集去评估
     */
    model.evaluate(tf.tensor(testData.sizeMB),tf.tensor(testData.timeSec)).print()

    // 保存模型
    model.save("file://predict1demo")
})();