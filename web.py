import streamlit as st
from PIL import Image

import argparse
import matplotlib.pyplot as plt

from colorizers import *

import time
parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Colorization")
st.write("")
st.write("")

file_up = st.file_uploader("Upload an image", type= ['png', 'jpg','jpeg'])
if file_up is None:
    image=Image.open("imgs/a.jpg")
    file_up="imgs/a.jpg"

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    #这一段使用s17模型
    # load colorizers
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if (opt.use_gpu):
        colorizer_siggraph17.cuda()
    img = load_img(file_up)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if (opt.use_gpu):
        tens_l_rs = tens_l_rs.cuda()
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    st.image(out_img_siggraph17, caption='Colorful Image.', use_column_width=True)

    st.success('successful')


    st.write("")

else:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    # 这一段使用s17模型
    # load colorizers
    colorizer_siggraph17 = siggraph17(pretrained=True).eval()
    if (opt.use_gpu):
        colorizer_siggraph17.cuda()
    img = load_img(file_up)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if (opt.use_gpu):
        tens_l_rs = tens_l_rs.cuda()
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    st.image(out_img_siggraph17, caption='Colorful Image.', use_column_width=True)

    st.success('successful')

    st.write("")
