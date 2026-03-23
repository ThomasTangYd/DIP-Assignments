import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    # warped_image = np.array(image)
    # ### FILL: Implement MLS or RBF based image warping
    if len(image.shape) == 2:
        height, width = image.shape
        is_gray = True
    else:
        height, width, ncolor = image.shape
        is_gray = False
        
    image_new = np.zeros(image.shape, dtype = np.float32)
    weight_sum = np.zeros((height, width), dtype = np.float32)

    for y in range(height):
        for x in range(width):
            v = np.array([x, y], dtype = np.float32)
            v_new = compute_affine(v, source_pts, target_pts, alpha, eps)

            x0 = int(np.floor(v_new[0]))
            y0 = int(np.floor(v_new[1]))
            x1 = x0 + 1
            y1 = y0 + 1

            if x0 < 0 or x1 >= width or y0 < 0 or y1 >= height:
                continue

            wx1 = v_new[0] - x0
            wy1 = v_new[1] - y0
            wx0 = 1.0 - wx1
            wy0 = 1.0 - wy1

            # if is_gray:
            #     color = image[y, x]
            #     image_new[y0, x0] += color * wx0 * wy0
            #     image_new[y0, x1] += color * wx1 * wy0
            #     image_new[y1, x0] += color * wx0 * wy1
            #     image_new[y1, x1] += color * wx1 * wy1
            # else:
            color = image[y, x, :]
            image_new[y0, x0, :] += color * wx0 * wy0
            image_new[y0, x1, :] += color * wx1 * wy0
            image_new[y1, x0, :] += color * wx0 * wy1
            image_new[y1, x1, :] += color * wx1 * wy1
            
            weight_sum[y0, x0] += wx0 * wy0
            weight_sum[y0, x1] += wx1 * wy0
            weight_sum[y1, x0] += wx0 * wy1
            weight_sum[y1, x1] += wx1 * wy1
    
    # if is_gray:
    #     image_new = np.divide(image_new, weight_sum + eps)
    # else:
    image_new = np.divide(image_new, weight_sum[:, :, np.newaxis] + eps)

    return np.clip(image_new, 0, 255).astype(np.uint8)

def compute_affine(v, source_pts, target_pts, alpha, eps):
    diff = source_pts - v
    w = 1.0 / (np.sum(diff ** 2, axis = 1) + eps) ** alpha
    sum_w = np.sum(w)
    if sum_w < eps:
        return v
    
    p_ast = np.sum(w[:, np.newaxis] * source_pts, axis = 0) / sum_w
    q_ast = np.sum(w[:, np.newaxis] * target_pts, axis = 0) / sum_w

    p_hat = source_pts - p_ast
    q_hat = target_pts - q_ast

    A = np.zeros((2, 2))
    B = np.zeros((2, 2))
    for i in range(len(w)):
        A += w[i] * np.outer(p_hat[i], p_hat[i])
        B += w[i] * np.outer(p_hat[i], q_hat[i])
    
    M = np.linalg.solve(A, B)
    return (v - p_ast).dot(M) + q_ast

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
