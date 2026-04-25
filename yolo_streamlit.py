# 马铃薯缺陷检测系统 - Streamlit 最终版
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 页面配置
st.set_page_config(
    page_title="马铃薯缺陷检测",
    page_icon="🥔",
    layout="wide"
)

# 标题
st.title("🥔 马铃薯缺陷检测系统")
st.subheader("基于YOLOv8的智能视觉检测")

# 加载模型（核心：路径和仓库一致）
@st.cache_resource  # 缓存模型，加速加载
def load_model():
    model = YOLO("best.pt")
    return model

# 加载模型
try:
    model = load_model()
    st.success("✅ 模型加载成功！")
except Exception as e:
    st.error(f"❌ 模型加载失败：{str(e)}")

# 上传图片
uploaded_file = st.file_uploader("请上传马铃薯图片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 打开图片
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    
    # 显示原图
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原图")
        st.image(img, use_column_width=True)
    
    # 预测
    with st.spinner("正在检测中..."):
        results = model(img_np)
    
    # 获取结果图
    res_img = results[0].plot()
    res_img = Image.fromarray(res_img[..., ::-1])
    
    # 显示结果
    with col2:
        st.subheader("检测结果")
        st.image(res_img, use_column_width=True)
    
    # 检测信息
    st.subheader("📊 检测详情")
    for r in results:
        if len(r.boxes) == 0:
            st.info("未检测到缺陷")
        else:
            st.success(f"检测到 {len(r.boxes)} 个目标")

st.markdown("---")
st.caption("✅ 基于YOLOv8 | Streamlit部署 | 马铃薯缺陷检测")
