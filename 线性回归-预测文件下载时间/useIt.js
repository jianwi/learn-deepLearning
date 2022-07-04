const tf = require("@tensorflow/tfjs-node")

async function run(){
    const model = await tf.loadLayersModel("file://predict1demo/model.json");
    // 打印模型的结构
    model.summary()
    // 预测结果
    model.predict(tf.tensor([100])).print()
}

run()