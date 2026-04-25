import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# 页面基础配置
st.set_page_config(page_title="YOLOv11马铃薯缺陷检测系统", layout="wide")

# 模型路径，和py文件同文件夹，无需修改
MODEL_PATH = "best.pt"

# 缓存模型，不用每次刷新都重新加载
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# 加载模型
try:
    model = load_model()
    st.success("✅ 模型加载成功！")
except Exception as e:
    st.error(f"❌ 模型加载失败：{e}")
    st.stop()

# 页面标题
st.title("基于YOLOv11的马铃薯缺陷检测系统")
st.divider()

# 侧边栏：参数设置+类别说明
with st.sidebar:
    st.header("检测参数设置")
    # 修正了参数名：maximum → max_value
    conf_threshold = st.slider("置信度阈值", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
    iou_threshold = st.slider("IOU阈值", min_value=0.1, max_value=1.0, value=0.45, step=0.05)
    st.divider()
    st.markdown("### 可检测病害类别")
    for idx, name in model.names.items():
        st.markdown(f"- {name}")

# 主界面：上传区
st.subheader("上传马铃薯图片")
# 支持批量上传多张图片
uploaded_files = st.file_uploader(
    "可选择单张/多张马铃薯图片", 
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True
)

# 全局统计
total_disease_count = {}

# 批量检测逻辑
if uploaded_files:
    st.divider()
    st.subheader("检测结果")
    
    # 遍历所有上传的图片，逐个检测
    for file in uploaded_files:
        # 读取图片
        img = Image.open(file)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 模型推理
        results = model.predict(
            source=img_cv,
            conf=conf_threshold,
            iou=iou_threshold,
            show_labels=True,
            show_conf=True,
            line_width=2
        )
        
        # 处理结果
        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        boxes = results[0].boxes

        # 单张图片的布局
        with st.container():
            st.markdown(f"#### 图片：{file.name}")
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="原图", use_container_width=True)
            with col2:
                st.image(result_img_rgb, caption="检测结果图", use_container_width=True)
            
            # 单张图片统计
            st.markdown("**当前图片检测统计**")
            if len(boxes) > 0:
                cls_names = model.names
                cls_count = {}
                for cls in boxes.cls:
                    cls_name = cls_names[int(cls)]
                    cls_count[cls_name] = cls_count.get(cls_name, 0) + 1
                    # 累计全局统计
                    total_disease_count[cls_name] = total_disease_count.get(cls_name, 0) + 1
                
                for name, count in cls_count.items():
                    st.write(f"- {name}：{count} 个")
            else:
                st.write("未检测到任何病害/目标")
            st.divider()

    # 全局统计结果
    if total_disease_count:
        st.subheader("📊 全部图片检测总统计")
        for name, count in total_disease_count.items():
            st.write(f"- {name}：总计 {count} 个")