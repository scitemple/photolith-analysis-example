# -*- coding: utf-8 -*-
"""
Created By: Rowan Temple
Created Date: 17/01/2025

Module contents ...
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage import measure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.transform import rescale

plt.ion()

# Load
fp = "./data/Photolith pattern.png"
rdat = plt.imread(fp)
# dat = np.asarray(dat)

# Display
print("Successfully laded TEM image. Dimension: ", rdat.shape)
plt.imshow(rdat)
# Binarise

# Convert to grayscale
dat = rgb2gray(rdat)

# Initial attempt to binarize
thresh = threshold_otsu(dat)
binary = (dat > thresh).astype(np.uint8)

# Plot histogram
plt.figure()
plt.hist(dat.ravel(), bins=256)
plt.title('Color istogram')
plt.axvline(thresh, color='r')

# Plot histogram again with threshold
plt.figure()
plt.hist(dat.ravel(), bins=256)
plt.title('Color histogram')
plt.axvline(thresh, color='r')


def create_image_comparison_figure():
    fig, axes = plt.subplots(ncols=2, figsize=(10, 3.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 2, 1)
    ax[1] = plt.subplot(1, 2, 2, sharex=ax[0], sharey=ax[0])
    [ax[i].axis('off') for i in range(2)]
    return ax


ax0, ax1 = create_image_comparison_figure()
ax0.imshow(dat, cmap="gray")
ax0.set_title('Original')
ax1.imshow(binary, cmap="gray")
ax1.set_title('Binarised')

# zoom in
ax0, ax1 = create_image_comparison_figure()
shpi, shpj = dat.shape
crop = (slice(round(shpi * 0.12), round(shpi * 0.18)),
        slice(round(shpj * 0.12), round(shpj * 0.18)))
ax0.imshow(dat[crop], cmap="gray")
ax0.set_title('Original')
ax1.imshow(binary[crop], cmap="gray")
ax0.set_title('Binarised')

# try again resized
scaling_factor = 3
dat = rescale(dat, scaling_factor)
binary = (dat > thresh).astype(np.uint8)

ax0, ax1 = create_image_comparison_figure()
shpi, shpj = dat.shape
crop = (slice(round(shpi * 0.12), round(shpi * 0.18)),
        slice(round(shpj * 0.12), round(shpj * 0.18)))
ax0.imshow(dat[crop], cmap="gray")
ax0.set_title('Original after rescale')
ax1.imshow(binary[crop], cmap="gray")
ax0.set_title('Binarised after rescale')

# Get image scale
coord = np.array([2115, 1959, 2127, 2151]) * scaling_factor
scale_line_y = 2118 * scaling_factor
scaleim = np.logical_not(binary[coord[0]:coord[2], coord[1]:coord[3]])
scale_lw = np.sum(scaleim[scale_line_y - coord[0], :])
pix_to_um = 50 / scale_lw
fig, ax = plt.subplots(1, 1)
ax.imshow(scaleim, cmap="gray")
print(f"The 50um scale line width is {round(scale_lw / scaling_factor)} pixels. "
      f"The total image size is {binary.shape[1] * pix_to_um:.3f} um x "
      f"{binary.shape[0] * pix_to_um:.3f} um")

# OK we've binarised the image ok. Now we want to do some analysis

# Label connected components and measure props
label_image = measure.label(binary)
properties = measure.regionprops(label_image)

df = {
    prop.label: {
        "device_label": "sample_1",
        "image": "Photolith pattern.png",
        "x_centre_pixels": prop.centroid[1] / scaling_factor,
        "y_centre_pixels": prop.centroid[0] / scaling_factor,
        "width_pix": (prop.bbox[3] - prop.bbox[1]) / scaling_factor,
        "height_pix": (prop.bbox[2] - prop.bbox[0]) / scaling_factor,
        "width": (prop.bbox[3] - prop.bbox[1]) * pix_to_um,
        "height": (prop.bbox[2] - prop.bbox[0]) * pix_to_um,
        "area": prop.area * pix_to_um ** 2,
        "circumference": prop.perimeter * pix_to_um
    } for prop in properties
}
df = pd.DataFrame().from_dict(df, orient="index")
print(f"Feature properties:")
print(df.iloc[:, 2:])


def plot_bboxes(ax, properties):
    for prop in properties:
        # Draw bounding box
        minr, minc, maxr, maxc = prop.bbox
        ax.add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                   edgecolor='blue', facecolor='none', lw=2))


xlims = round(0.3 * binary.shape[1]), round(0.7 * binary.shape[1])
ylims = round(0.7 * binary.shape[0]), round(0.3 * binary.shape[0])

# Plot widths
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax0, ax1 = ax.ravel()
ax0.imshow(binary, cmap="gray")
plt.suptitle("Feature width")
plot_bboxes(ax0, properties)
for prop in properties:
    # Annotate particle properties
    ax0.text(prop.centroid[1], prop.bbox[0],
             f"{(prop.bbox[3] - prop.bbox[1]) * pix_to_um:.3f} um",
             color='red', fontsize=10, ha='center', va='top')

ax1.imshow(binary, cmap="gray")
plot_bboxes(ax1, properties)
for prop in properties:
    x, y = prop.centroid[1], prop.centroid[0]
    if (xlims[0] <= x <= xlims[1]) and (ylims[1] <= y <= ylims[0]):
        ax1.text(prop.centroid[1], prop.bbox[0],
                 f"{(prop.bbox[3] - prop.bbox[1]) * pix_to_um:.3f} um",
                 color='red', fontsize=10, ha='center', va='top')
ax1.set_xlim(*xlims)
ax1.set_ylim(*ylims)

# Plot heights
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax0, ax1 = ax.ravel()
ax0.imshow(binary, cmap="gray")
plt.suptitle("Feature height")
plot_bboxes(ax0, properties)
for prop in properties:
    # Annotate particle properties
    ax0.text(prop.centroid[1], prop.bbox[0],
             f"{(prop.bbox[2] - prop.bbox[0]) * pix_to_um:.3f} um",
             color='red', fontsize=10, ha='center', va='top')

ax1.imshow(binary, cmap="gray")
plot_bboxes(ax1, properties)
for prop in properties:
    x, y = prop.centroid[1], prop.centroid[0]
    if (xlims[0] <= x <= xlims[1]) and (ylims[1] <= y <= ylims[0]):
        ax1.text(prop.centroid[1], prop.bbox[0],
                 f"{(prop.bbox[2] - prop.bbox[0]) * pix_to_um:.3f} um",
                 color='red', fontsize=10, ha='center', va='top')
ax1.set_xlim(*xlims)
ax1.set_ylim(*ylims)

# Plot labels
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax0, ax1 = ax.ravel()
ax0.imshow(binary, cmap="gray")
plt.suptitle("Feature label")
plot_bboxes(ax0, properties)
for prop in properties:
    # Annotate particle properties
    ax0.text(prop.centroid[1], prop.centroid[0], f"{prop.label}",
             color='red', fontsize=10, ha='center', va='top')

ax1.imshow(binary, cmap="gray")
plot_bboxes(ax1, properties)

for prop in properties:
    x, y = prop.centroid[1], prop.centroid[0]
    if (xlims[0] <= x <= xlims[1]) and (ylims[1] <= y <= ylims[0]):
        ax1.text(x, y, f"{prop.label}",
                 color='red', fontsize=10, ha='center', va='top')
ax1.set_xlim(*xlims)
ax1.set_ylim(*ylims)

df = df.drop(37)

label_map = {
    'vertical_left': [3, 6, 9, 12, 15, 20, 21, 25, 27],
    'vertical_right': [2, 5, 8, 11, 14, 17, 19, 23, 26],
    'horizontal_top': [1, 4, 7, 10, 13, 16, 18, 22, 24],
    'horizontal_bottom': [36, 35, 34, 33, 32, 31, 30, 29, 28]
}

expected_width = [50, 30, 20, 10, 5, 3, 2, 1, 0.5]


for idx, row in df.iterrows():
    bar_type = [k for k, v in label_map.items() if idx in v][0]
    expected_bar_width = expected_width[label_map[bar_type].index(idx)]
    actual_bar_width = row["width"] if row["width"] < row["height"] else row["height"]
    df.loc[idx, "bar_type"] = bar_type
    df.loc[idx, "expected_bar_width"] = expected_bar_width
    df.loc[idx, "actual_bar_width"] = actual_bar_width

fig, ax = plt.subplots(1, 1)
sub1 = df.loc[df["bar_type"].str.startswith("horizontal")]
sub2 = df.loc[df["bar_type"].str.startswith("vertical")]
plt.plot(sub1["expected_bar_width"], sub1["actual_bar_width"] / sub1["expected_bar_width"], "kx", label="Horizontal patterned bars")
plt.plot(sub2["expected_bar_width"], sub2["actual_bar_width"] / sub2["expected_bar_width"], "gx", label="Vertical patterned bars")
plt.xlabel("Expected bar width (um)")
plt.ylabel("Actual / Expected width")
plt.legend()

fig, ax = plt.subplots(1, 1)
sub1 = df.loc[df["bar_type"].str.startswith("horizontal")]
sub2 = df.loc[df["bar_type"].str.startswith("vertical")]
plt.plot(sub1["expected_bar_width"], sub1["actual_bar_width"] - sub1["expected_bar_width"], "kx", label="Horizontal patterned bars")
plt.plot(sub2["expected_bar_width"], sub2["actual_bar_width"] - sub2["expected_bar_width"], "gx", label="Vertical patterned bars")
plt.xlabel("Expected bar width (um)")
plt.ylabel("Actual - Expected width")
plt.legend()

