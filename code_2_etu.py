"""
Compute mean value in multiple ROIs

This is the code for this TP to be completed and returned to your teacher.
The blanks shown as ( * * * ) need to be replaced with the proper coding.
Good luck
"""
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#%% functions (complete functions below)

def attenuation(Im1, White):
    """
    Attenuation function (Beer-Lambert law)

    Args:
        Im1 (ndarray): Flat-field corrected image.
        White (ndarray): White/flat-field reference image.

    Returns:
        TYPE: attenuation image.
    """
    # 衰减 = -ln(I/I0)，根据比尔-朗伯定律
    A = -np.log(Im1 / White)
    return A


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

#%%
# Read image and perform flat-field

black = mpimg.imread("Black.tif")      # 暗场图像
white = mpimg.imread("White.tif")      # 平场参考图像（无样品）
data = "Q11.tif"                       # 纸板阶梯图像
im = mpimg.imread(data)
im1 = flat_field(im, black, white)
nGradins = 7                           # 阶梯数量（根据实际样品调整） 

# Define ROI coordinates (/!\ Update coordinates)
x = 100
y = [10, 100, 180, 250, 340, 400, 500, 550]; # top left corner	
width = 50
height = 30

plt.plot(x, y[0], marker='v', color="white") 
plt.imshow(im1,cmap='gray') 
plt.show()

#%% Compute attenuation
Im2 = 'White.tif'
White = mpimg.imread(Im2)
# 对White也进行暗场校正
White_corrected = White - black
Q12 = attenuation(im1, White_corrected)
plt.imshow(Q12, cmap='gray')
plt.colorbar()
plt.title('Attenuation image')
plt.show()

# 保存衰减图像
plt.imsave('Q12.tif', Q12, cmap='gray')

#%% Compute mean value in each of the rois
M = np.empty(1, dtype=object)
for i in range(0, nGradins):
    Roi = im1[y[i]:y[i]+height, x:x+width]
    Mean = np.mean(Roi)
    print('Mean ROI N°', i, 'is', Mean)
    M = np.vstack((M, Mean))

M = M[1:nGradins+1]
#%% Display all the ROIs
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3,3)
fig.suptitle('Vertically stacked subplots')

ax1.imshow(im1[y[0]:y[0]+height, x:x+width],cmap='gray')
ax2.imshow(im1[y[1]:y[1]+height, x:x+width],cmap='gray')
ax3.imshow(im1[y[2]:y[2]+height, x:x+width],cmap='gray')
ax4.imshow(im1[y[3]:y[3]+height, x:x+width],cmap='gray')
ax5.imshow(im1[y[4]:y[4]+height, x:x+width],cmap='gray')
ax6.imshow(im1[y[5]:y[5]+height, x:x+width],cmap='gray')
ax7.imshow(im1[y[6]:y[6]+height, x:x+width],cmap='gray')
ax8.imshow(im1[y[7]:y[7]+height, x:x+width],cmap='gray')


#%% Plot the mean values
# plot
X = [1,2,3,4,5,6,7]
fig, ax = plt.subplots()
ax.plot(X, M, linewidth=2.0)
plt.show()

#%% Compute attenuation values of paperboard only
MA = np.empty(1, dtype=object)
for i in range(0, nGradins):
    Roi = Q12[y[i]:y[i]+height, x:x+width]
    Mean = np.mean(Roi)
    print('Mean Attenuation ROI N°', i, 'is', Mean)
    MA = np.vstack((MA, Mean))

MA = MA[1:nGradins+1]

#%%  Plot attenuation values of paperboard only
fig, ax = plt.subplots()
ax.plot(X, MA, 'o-', linewidth=2.0)
ax.set_xlabel('Step number (thickness)')
ax.set_ylabel('Attenuation')
ax.set_title('Attenuation vs Thickness (Beer-Lambert Law)')
ax.grid(True)
plt.show()

# 保存图表
fig.savefig('Q12_plot.tif')
print("Plot saved as Q12_plot.tif")

