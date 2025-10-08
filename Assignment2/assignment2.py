import cv2
import matplotlib.pyplot as plt

# Read the images
im1 = cv2.imread("images/board-157165_1280.png")
im2 = cv2.imread("images/jellyfish-698521_1280.jpg")
im3 = cv2.imread("images/outdoors-5129182_1280.jpg")

# Convert to grayscale
# OpenCV loads images in BGR format, so convert before gray if needed
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
im3_gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

# Convert to double precision (0–1 range)
im1_gray = im1_gray.astype("float64") / 255.0
im2_gray = im2_gray.astype("float64") / 255.0
im3_gray = im3_gray.astype("float64") / 255.0

# Plot the images
plt.figure()
plt.imshow(im1_gray, cmap="gray")
plt.title("Image 1 (Chessboard)")
plt.axis("off")

plt.figure()
plt.imshow(im2_gray, cmap="gray")
plt.title("Image 2 (Jellyfish)")
plt.axis("off")

plt.figure()
plt.imshow(im3_gray, cmap="gray")
plt.title("Image 3 (New York Skyline)")
plt.axis("off")

plt.show()