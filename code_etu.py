"""
This is the code for this TP to be completed and returned to your teacher.
The blanks shown as ( * * * ) need to be replaced with the proper coding.
Good luck
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# %% functions (complete functions below)

def my_function(im1, im2):
    """
    Read noise correction

    Args:
        im1 (ndarray): Raw image.
        im2 (ndarray): Black image.

    Returns:
        TYPE: Corrected image.
    """

    return im1 - im2
def flat_field(im1, im2, im3):
    """
    Flat field correction

    Args:
        im1 (ndarray): Raw image.
        im2 (ndarray): Black image.
        im3 (ndarray): White image.

    Returns:
        TYPE: Flat-field corrected image.
    """

    return (im1 - im2) / (im3 - im2)

# %%
## (Q1) Read and show the image
my_image = 'Q1.tif'
im1 = mpimg.imread(my_image)
plt.imshow(im1)
plt.show()

## (Q4) Show pixel value
x, y = 150, 59
print(im1[x, y])

# %%
## (Q6) Correction method / You need to complete function "my_function"
Black = 'Q5_dark.tif'  # 暗场图像（X射线关闭时采集）
im2 = mpimg.imread(Black)
im_corrected = my_function(im1, im2)
plt.imshow(im_corrected)
plt.show()

## (Q7) Verification of correction method
# 读取原始图像
im_1000 = mpimg.imread('Q3_1000.tif')
im_4000 = mpimg.imread('Q3_4000.tif')
im_dark = mpimg.imread('Q5_dark.tif')  # 暗场图像

# 暗电流校正
Q7_1000ms = my_function(im_1000, im_dark)
Q7_4000ms = my_function(im_4000, im_dark)

# 计算比值，验证线性关系（应该接近4）
Resultat = Q7_4000ms / Q7_1000ms
plt.imshow(Resultat)
plt.colorbar()
plt.title('Ratio 4000ms/1000ms (should be ~4)')
plt.show()

# 保存校正后的图像
plt.imsave('Q7_1000ms.tif', Q7_1000ms)
plt.imsave('Q7_4000ms.tif', Q7_4000ms)
plt.imsave('Q7_div.tif', Resultat)

## (Q9-Q10) Flat field method / You need to complete function "flat_field"
# 读取图像
im_raw = mpimg.imread('Q3_1000.tif')       # 原始图像（有样品）
im_dark = mpimg.imread('Q5_dark.tif')       # 暗场图像（X射线关闭）
im_white = mpimg.imread('Q9_1000ms.tif')    # 平场参考图像（无样品，X射线开启）

# 平场校正
Q10_1000ms = flat_field(im_raw, im_dark, im_white)
plt.imshow(Q10_1000ms, cmap='gray')
plt.colorbar()
plt.title('Flat-field corrected image (1000ms)')
plt.show()

# 保存平场校正后的图像
plt.imsave('Q10_1000ms.tif', Q10_1000ms, cmap='gray')

# 对2000ms图像也进行平场校正
im_raw_2000 = mpimg.imread('Q3_2000.tif')
im_white_2000 = mpimg.imread('Q9_2000ms.tif')  # 需要采集对应的平场参考
Q10_2000ms = flat_field(im_raw_2000, im_dark, im_white_2000)
plt.imsave('Q10_2000ms.tif', Q10_2000ms, cmap='gray')
