import gradio as gr
from ultralytics import YOLO
from PIL import Image

# 模型路径，和py文件同文件夹，无需修改
MODEL_PATH = "best.pt"

# 加载YOLOv11模型
try:
    model = YOLO(MODEL_PATH)
    print("✅ 模型加载成功！可识别类别：", model.names)
except Exception as e:
    print("❌ 模型加载失败，请检查best.pt文件是否和py文件在同一个文件夹！错误信息：", e)
    exit()

# 核心检测函数
def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        line_width=2
    )
    # 处理结果图适配网页展示
    result_img = results[0].plot()
    result_img_pil = Image.fromarray(result_img[..., ::-1])
    return result_img_pil

# 搭建网页界面（适配Gradio 4.x）
with gr.Blocks(title="YOLOv11马铃薯缺陷检测系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 基于YOLOv11的马铃薯缺陷检测系统")
    gr.Markdown("支持检测马铃薯黑斑、褐腐、干腐、软腐四类病害，上传图片即可一键检测")
    
    with gr.Row():
        # 左侧输入区
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="请上传马铃薯图片")
            conf_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.25, step=0.05,
                label="置信度阈值（数值越高，检测越严格）"
            )
            iou_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.45, step=0.05,
                label="IOU阈值（数值越高，重复框越少）"
            )
            predict_btn = gr.Button("开始检测", variant="primary", size="lg")

        # 右侧结果区
        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="检测结果图")

    # 绑定按钮和检测函数
    predict_btn.click(
        fn=predict_image,
        inputs=[input_img, conf_slider, iou_slider],
        outputs=output_img
    )

# 启动网页
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
    # 给老师发公网链接，就注释上面一行，打开下面这行
    # demo.launch(share=True)python yolo_gradio_demo.py